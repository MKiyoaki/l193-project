import os
import gc
import json
import argparse
from itertools import islice
import torch as t
from torch import nn
from tqdm import tqdm
from nnsight import LanguageModel
from attribution import patching_effect
from dictionary_loading_utils import load_saes_and_submodules
from pathlib import Path

from utils.sfc_utils import save_extracted_features_to_json
from utils.data_utils import get_dataset_path
from utils.data_loader import SycophancyDataLoader
from utils.configs_utils import (
    DTYPE, DEVICE, BATCH_SIZE, N_BATCHES, EFFECTS_DIR
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class IdentityDict(nn.Module):
    """
    Dummy dictionary to bypass SAE logic and compute IG directly on dense neurons.
    Must implement encode and decode to satisfy the patching_effect interface.
    """

    def __init__(self, d_model):
        super().__init__()
        self.dict_size = d_model
        self.d_model = d_model

    def forward(self, x):
        return x

    def encode(self, x):
        return x

    def decode(self, f):
        return f


def metric_fn(model, fac_ids, syc_ids):
    """Compute difference between sycophantic and factual logits."""
    logits = model.lm_head.output
    last_token_logits = logits[:, -1, :]
    batch_indices = t.arange(last_token_logits.size(0), device=DEVICE)
    syc_logits = last_token_logits[batch_indices, syc_ids]
    fac_logits = last_token_logits[batch_indices, fac_ids]
    return syc_logits - fac_logits


def get_submodules(model, start_layer, end_layer):
    """Retrieve residual stream and MLP submodules for the dense model."""
    submodules = []
    for layer_idx in range(start_layer, end_layer + 1):
        res_submod = model.model.layers[layer_idx]
        res_submod.name = f"resid_{layer_idx}"
        submodules.append(res_submod)

        mlp_submod = model.model.layers[layer_idx].mlp
        mlp_submod.name = f"mlp_{layer_idx}"
        submodules.append(mlp_submod)

    return submodules


def evaluate_completeness_thresholds(
    model,
    dataloader,
    submodules,
    global_aggregated_effects,
    thresholds,
    d_model,
    device,
    tracer_kwargs
):
    """Evaluate sycophancy rate using mean ablation via out-of-place math based on effect thresholds."""
    batches = list(dataloader)
    curve_results = {}

    for thresh in thresholds:
        print(f"\nEvaluating mean ablation for Threshold > {thresh}...")

        ablation_masks = {
            submod.name: t.zeros(d_model, dtype=t.float32, device=device)
            for submod in submodules
        }
        num_neurons_ablated = 0

        for submod in submodules:
            effects = global_aggregated_effects[submod.name]
            ablate_indices = t.where(effects > thresh)[0]
            num_features = len(ablate_indices)

            if num_features > 0:
                ablation_masks[submod.name][ablate_indices] = 1.0
                num_neurons_ablated += num_features

        if num_neurons_ablated == 0:
            print(f"No neurons found above threshold {thresh}. Skipping.")
            continue

        syc_logits_list = []
        fac_logits_list = []

        with tqdm(total=len(batches), desc=f"Eval ({num_neurons_ablated} neurons)") as pbar:
            for text_batch, fac_ids, syc_ids in batches:
                fac_ids_dev = fac_ids.to(device)
                syc_ids_dev = syc_ids.to(device)

                with model.trace(text_batch, **tracer_kwargs):
                    for submod in submodules:
                        output_proxy = submod.output

                        if hasattr(output_proxy, "shape"):
                            act = output_proxy
                            is_tuple = False
                        else:
                            act = output_proxy[0]
                            is_tuple = True

                        mean_act = act.mean(dim=(0, 1), keepdim=True)
                        mask_f = ablation_masks[submod.name].to(act.dtype)
                        ablated_act = act * (1.0 - mask_f) + mean_act * mask_f

                        if is_tuple:
                            submod.output = (ablated_act,) + output_proxy[1:]
                        else:
                            submod.output = ablated_act

                    logits = model.lm_head.output
                    last_token_logits = logits[:, -1, :]

                    batch_indices = t.arange(
                        last_token_logits.size(0), device=device)

                    s_proxy = last_token_logits[batch_indices, syc_ids_dev].save(
                    )
                    f_proxy = last_token_logits[batch_indices, fac_ids_dev].save(
                    )

                syc_logits_list.append(s_proxy.value.detach().cpu())
                fac_logits_list.append(f_proxy.value.detach().cpu())
                pbar.update(1)

        all_syc = t.cat(syc_logits_list)
        all_fac = t.cat(fac_logits_list)

        syc_rate = (all_syc > all_fac).float().mean().item()
        curve_results[num_neurons_ablated] = syc_rate
        print(
            f"Result for Threshold {thresh} ({num_neurons_ablated} Neurons Ablated): Sycophancy Rate = {syc_rate:.4%}")

    return curve_results


def run_dense_experiment(args):
    """Main execution pipeline for dense baseline experiment."""
    os.makedirs(EFFECTS_DIR, exist_ok=True)
    tracer_kwargs = dict(scan=False, validate=False)
    thresholds = args.thresholds

    model_name = "google/gemma-2-2b"
    total_layers = 26
    d_model = 2304

    print(f"Loading Model for Dense Baseline: {model_name}...")
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

    print("Starting chunked dense scoring...")
    for start_layer in range(0, total_layers, CHUNK_SIZE):
        end_layer = min(start_layer + CHUNK_SIZE - 1, total_layers - 1)

        submodules, sae_dicts = load_saes_and_submodules(
            model,
            start_layer=start_layer,
            thru_layer=end_layer,
            include_embed=False,
            dtype=DTYPE,
            device=DEVICE,
        )

        dictionaries = {submod: IdentityDict(
            d_model).to(DEVICE) for submod in submodules}
        chunk_aggregated_effects = {submod.name: 0 for submod in submodules}

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
                else:
                    chunk_aggregated_effects[submodule.name] += tensor_data.to(
                        "cpu")

            del raw_effects, _
            gc.collect()

        global_aggregated_effects.update(chunk_aggregated_effects)
        del submodules, dictionaries, chunk_aggregated_effects
        gc.collect()

    total_examples = BATCH_SIZE * N_BATCHES
    for k in global_aggregated_effects:
        global_aggregated_effects[k] /= total_examples

    min_eval_thresh = min(thresholds)
    dense_neurons_path = Path(
        EFFECTS_DIR
    ) / f"extracted_dense_neurons_{args.experiment_name}.json"

    save_extracted_features_to_json(
        global_aggregated_effects=global_aggregated_effects,
        output_path=dense_neurons_path,
        min_threshold=min_eval_thresh
    )

    print("\n--- Starting Dense Neuron Completeness Evaluation ---")

    eval_batches = islice(dataloader.get_text_batches(
        split="test", batch_size=BATCH_SIZE), N_BATCHES)

    all_submodules = get_submodules(model, 0, total_layers - 1)

    curve_results = evaluate_completeness_thresholds(
        model=model,
        dataloader=eval_batches,
        submodules=all_submodules,
        global_aggregated_effects=global_aggregated_effects,
        thresholds=thresholds,
        d_model=d_model,
        device=DEVICE,
        tracer_kwargs=tracer_kwargs
    )

    output_data = {
        "method": "Dense_Neurons_Threshold",
        "data": curve_results
    }

    output_path = Path(EFFECTS_DIR) / \
        f"dense_completeness_results_{args.experiment_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"\nDense Completeness Curve results successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        help="List of thresholds to evaluate dense neuron completeness."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        choices=["nlp", "phi", "political"],
        default="nlp",
    )
    args = parser.parse_args()
    run_dense_experiment(args)


if __name__ == "__main__":
    main()
