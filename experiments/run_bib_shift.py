import os
import gc
import hashlib
import argparse
from itertools import islice
import torch as t
from tqdm import tqdm
from nnsight import LanguageModel
from attribution import patching_effect
from dictionary_loading_utils import load_saes_and_submodules
from utils.data_loader import SycophancyDataLoader
from utils.configs_utils import (
    DTYPE, DEVICE, BATCH_SIZE, N_BATCHES,
    EFFECTS_DIR, TOP_K_TO_ABLATE
)

from utils.sfc_utils import (
    print_top_features_and_links,
    generate_ablation_blacklist,
    prepare_ablation_masks,
    run_evaluation
)

# Set environment variable to reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def metric_fn(model, fac_ids, syc_ids):
    # Computes logit diff between sycophantic and factual tokens
    logits = model.lm_head.output
    last_token_logits = logits[:, -1, :]
    batch_indices = t.arange(last_token_logits.size(0), device=DEVICE)
    syc_logits = last_token_logits[batch_indices, syc_ids]
    fac_logits = last_token_logits[batch_indices, fac_ids]
    return syc_logits - fac_logits


def run_experiment(args):
    os.makedirs(EFFECTS_DIR, exist_ok=True)
    tracer_kwargs = dict(scan=False, validate=False)

    # Determine the model name and total layers based on parsed arguments
    if args.model == "gemma":
        model_name = "google/gemma-2-2b"
        total_layers = 26
    elif args.model == "pythia":
        model_name = "EleutherAI/pythia-70m-deduped"
        total_layers = 6
    else:
        raise ValueError("Unsupported model type.")

    print(f"Loading Model: {model_name}...")
    model = LanguageModel(
        model_name,
        device_map=DEVICE,
        dispatch=True,
        attn_implementation="eager",
        dtype=DTYPE
    )

    # Freeze all model parameters to save gradient memory overhead
    model.requires_grad_(False)

    dataloader = SycophancyDataLoader()
    # Convert generator to list to reuse the same data across multiple chunks
    batches = list(islice(dataloader.get_text_batches(
        split="train", batch_size=BATCH_SIZE), N_BATCHES))

    global_aggregated_effects = {}
    CHUNK_SIZE = 6

    print("Starting chunked scoring to prevent OOM...")
    for start_layer in range(0, total_layers, CHUNK_SIZE):
        end_layer = min(start_layer + CHUNK_SIZE - 1, total_layers - 1)
        print(
            f"\n--- Loading SAEs for layers {start_layer} to {end_layer} ---")

        submodules, dictionaries = load_saes_and_submodules(
            model,
            start_layer=start_layer,
            thru_layer=end_layer,
            include_embed=False,
            dtype=DTYPE,
            device=DEVICE,
        )

        chunk_aggregated_effects = {
            submodule.name: 0 for submodule in submodules}
        submodule_names = [s.name for s in submodules]

        for clean, fac_ids, syc_ids in tqdm(batches, desc=f"Scoring L{start_layer}-L{end_layer}"):
            # We do not use disk cache for raw batch effects anymore!
            # Raw SAE IG tensors are too massive (hundreds of MBs per batch).
            t.cuda.empty_cache()

            # Execute IG with reduced steps to minimize activation memory footprint
            raw_effects, *_ = patching_effect(
                clean,
                None,
                model,
                submodules,
                dictionaries,
                metric_fn,
                steps=5,
                metric_kwargs=dict(fac_ids=fac_ids, syc_ids=syc_ids),
                method='ig'
            )

            # Aggregate IMMEDIATELY in memory to reduce dimensional footprint from
            # [Batch, Seq, Feature] to just [Feature].
            for submodule in submodules:
                tensor_obj = raw_effects[submodule.name]

                # Extract the actual tensor
                if hasattr(tensor_obj, 'act'):
                    tensor_data = tensor_obj.act
                else:
                    tensor_data = tensor_obj

                # Dynamically handle different tensor dimensions and accumulate directly
                if tensor_data.ndim == 3:
                    chunk_aggregated_effects[submodule.name] += (
                        tensor_data[:, 1:, :]
                    ).sum(dim=1).sum(dim=0).to("cpu")
                elif tensor_data.ndim == 2:
                    chunk_aggregated_effects[submodule.name] += tensor_data.sum(
                        dim=0).to("cpu")
                elif tensor_data.ndim == 1:
                    chunk_aggregated_effects[submodule.name] += tensor_data.to(
                        "cpu")
                else:
                    raise ValueError(
                        f"Unexpected tensor dimension: {tensor_data.ndim}")

            # Delete the massive raw tensor and force garbage collection immediately
            del raw_effects, _
            gc.collect()
            t.cuda.empty_cache()

        # Merge chunk data into the global dictionary
        global_aggregated_effects.update(chunk_aggregated_effects)

        # Forcefully clear VRAM before the next chunk
        del submodules
        del dictionaries
        del chunk_aggregated_effects
        gc.collect()
        t.cuda.empty_cache()

    total_examples = BATCH_SIZE * N_BATCHES
    global_aggregated_effects = {
        k: v / total_examples for k, v in global_aggregated_effects.items()
    }

    print("\nExtracting Global Top Features...")
    top_features = []
    for submod_name, effects_tensor in global_aggregated_effects.items():
        top_vals, top_idxs = t.topk(effects_tensor, k=5)
        for val, idx in zip(top_vals, top_idxs):
            if val > 0:
                top_features.append((val.item(), submod_name, idx.item()))

    top_features.sort(key=lambda x: x[0], reverse=True)
    print_top_features_and_links(top_features)

    feats_to_ablate = generate_ablation_blacklist(
        top_features, TOP_K_TO_ABLATE, num_layers=total_layers)

    # Determine which SAE layers actually need to be reloaded for evaluation
    layers_to_reload = set()
    for submod_name, feats in feats_to_ablate.items():
        if len(feats) > 0 and "_" in submod_name:
            layers_to_reload.add(int(submod_name.split("_")[1]))

    if not layers_to_reload:
        print("No features to ablate. Exiting.")
        return

    eval_start_layer = min(layers_to_reload)
    eval_end_layer = max(layers_to_reload)

    print(
        f"\nReloading SAEs for evaluation (Layers {eval_start_layer} to {eval_end_layer})...")
    submodules, dictionaries = load_saes_and_submodules(
        model,
        start_layer=eval_start_layer,
        thru_layer=eval_end_layer,
        include_embed=False,
        dtype=DTYPE,
        device=DEVICE,
    )

    print("Preparing ablation masks...")
    ablation_masks = prepare_ablation_masks(
        feats_to_ablate,
        submodules,
        dictionaries,
        DEVICE
    )

    run_evaluation(
        model,
        dataloader,
        submodules,
        dictionaries,
        ablation_masks,
        tracer_kwargs
    )


def main():
    # Setup argparse to control model selection from the command line
    parser = argparse.ArgumentParser(
        description="Run Feature Circuits Experiment")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma", "pythia"],
        default="gemma",
        help="Choose which model to load: 'gemma' or 'pythia'"
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
