import os
import gc
import json
import argparse
from itertools import islice
import torch as t
from tqdm import tqdm
from nnsight import LanguageModel
from attribution import patching_effect
from pathlib import Path

from dictionary_loading_utils import load_saes_and_submodules
from utils.data_utils import get_dataset_path

from utils.data_loader import SycophancyDataLoader
from utils.configs_utils import (
    DTYPE, DEVICE, BATCH_SIZE, N_BATCHES, EFFECTS_DIR
)
from utils.sfc_utils import (
    prepare_ablation_masks,
    run_evaluation,
    save_extracted_features_to_json
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def metric_fn(model, fac_ids, syc_ids):
    """Compute difference between sycophantic and factual logits."""
    logits = model.lm_head.output
    last_token_logits = logits[:, -1, :]
    batch_indices = t.arange(last_token_logits.size(0), device=DEVICE)
    syc_logits = last_token_logits[batch_indices, syc_ids]
    fac_logits = last_token_logits[batch_indices, fac_ids]
    return syc_logits - fac_logits


def evaluate_completeness_thresholds(
    model,
    dataloader,
    global_aggregated_effects,
    thresholds,
    total_layers,
    chunk_size,
    tracer_kwargs
):
    """Evaluate completeness by dynamically ablating features above given thresholds."""
    completeness_curve_results = {}
    print("\n--- Starting Chunked Completeness Evaluation (Threshold-Based) ---")

    for start_layer in range(0, total_layers, chunk_size):
        end_layer = min(start_layer + chunk_size - 1, total_layers - 1)
        chunk_key = f"L{start_layer}-L{end_layer}"
        completeness_curve_results[chunk_key] = {}

        # Filter global keys for current chunk
        expected_submods = [
            name for name in global_aggregated_effects.keys()
            if start_layer <= int(name.split("_")[1]) <= end_layer
        ]

        # Pre-check: Does the chunk have ANY features above the minimum threshold?
        min_thresh = min(thresholds)
        chunk_has_features_overall = any(
            (global_aggregated_effects[submod] > min_thresh).any().item()
            for submod in expected_submods
        )

        if not chunk_has_features_overall:
            continue

        print(f"\nReloading SAEs for chunk {chunk_key}...")
        submodules, dictionaries = load_saes_and_submodules(
            model,
            start_layer=start_layer,
            thru_layer=end_layer,
            include_embed=False,
            dtype=DTYPE,
            device=DEVICE,
        )

        for thresh in thresholds:
            print(f"\nEvaluating Threshold > {thresh} for {chunk_key}...")

            chunk_feats_to_ablate = {}
            chunk_has_features_for_this_thresh = False

            for submod in submodules:
                effects = global_aggregated_effects[submod.name]
                # Filter features by threshold
                ablate_indices = t.where(effects > thresh)[0].tolist()
                chunk_feats_to_ablate[submod.name] = ablate_indices

                if len(ablate_indices) > 0:
                    chunk_has_features_for_this_thresh = True

            if not chunk_has_features_for_this_thresh:
                continue

            # Calculate actual feature count for JSON key compatibility
            num_features_ablated = sum(len(indices)
                                       for indices in chunk_feats_to_ablate.values())

            ablation_masks = prepare_ablation_masks(
                chunk_feats_to_ablate,
                submodules,
                dictionaries,
                DEVICE
            )

            clean_score, ablated_score = run_evaluation(
                model,
                dataloader,
                submodules,
                dictionaries,
                ablation_masks,
                tracer_kwargs
            )

            # Store result using the exact number of ablated features
            completeness_curve_results[chunk_key][num_features_ablated] = ablated_score / 100.0

        del submodules
        del dictionaries
        if 'ablation_masks' in locals():
            del ablation_masks
        gc.collect()
        t.cuda.empty_cache()

    return completeness_curve_results


def run_experiment(args):
    """Main experiment runner."""
    os.makedirs(EFFECTS_DIR, exist_ok=True)
    tracer_kwargs = dict(scan=False, validate=False)

    thresholds = args.thresholds

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
    model.requires_grad_(False)

    dataloader = SycophancyDataLoader(
        file_path=get_dataset_path(args.experiment_name)
    )
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

        for clean, fac_ids, syc_ids in tqdm(batches, desc=f"Scoring L{start_layer}-L{end_layer}"):
            t.cuda.empty_cache()

            raw_effects, *_ = patching_effect(
                clean, None, model, submodules, dictionaries, metric_fn,
                steps=5, metric_kwargs=dict(fac_ids=fac_ids, syc_ids=syc_ids), method='ig'
            )

            for submodule in submodules:
                tensor_obj = raw_effects[submodule]
                tensor_data = tensor_obj.act if hasattr(
                    tensor_obj, 'act') else tensor_obj

                if tensor_data.ndim == 3:
                    chunk_aggregated_effects[submodule.name] += (
                        tensor_data[:, 1:, :]).sum(dim=1).sum(dim=0).to("cpu")
                elif tensor_data.ndim == 2:
                    chunk_aggregated_effects[submodule.name] += tensor_data.sum(
                        dim=0).to("cpu")
                elif tensor_data.ndim == 1:
                    chunk_aggregated_effects[submodule.name] += tensor_data.to(
                        "cpu")
                else:
                    raise ValueError(
                        f"Unexpected tensor dimension: {tensor_data.ndim}")

            del raw_effects, _
            gc.collect()
            t.cuda.empty_cache()

        global_aggregated_effects.update(chunk_aggregated_effects)

        del submodules, dictionaries, chunk_aggregated_effects
        gc.collect()
        t.cuda.empty_cache()

    # Normalize by total examples
    total_examples = BATCH_SIZE * N_BATCHES
    for k in global_aggregated_effects:
        global_aggregated_effects[k] /= total_examples

    # We use the minimum threshold from our evaluation list to filter out noise
    min_eval_thresh = min(thresholds)
    sfc_features_path = Path(EFFECTS_DIR) / \
        f"extracted_sfc_features_{args.experiment_name}.json"

    save_extracted_features_to_json(
        global_aggregated_effects=global_aggregated_effects,
        output_path=sfc_features_path,
        min_threshold=min_eval_thresh
    )

    # Call the new helper function
    completeness_curve_results = evaluate_completeness_thresholds(
        model=model,
        dataloader=dataloader,
        global_aggregated_effects=global_aggregated_effects,
        thresholds=thresholds,
        total_layers=total_layers,
        chunk_size=CHUNK_SIZE,
        tracer_kwargs=tracer_kwargs
    )

    output_data = {
        "method": "SAE_Features_Threshold",
        "data": completeness_curve_results
    }

    output_path = Path(EFFECTS_DIR) / \
        f"sfc_completeness_results_{args.experiment_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"\nSFC Completeness Curve results successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Feature Circuits Experiment")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma", "pythia"],
        default="gemma",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        choices=["nlp", "phi", "political"],
        default="nlp",
    )
    # Define thresholds directly via command line
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.05, 0.01, 0.005, 0.001],
        help="List of thresholds to evaluate feature completeness."
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
