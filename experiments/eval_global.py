import os
import gc
import json
import argparse
from collections import defaultdict
import torch as t
from torch import nn
from nnsight import LanguageModel
from pathlib import Path

from dictionary_loading_utils import load_saes_and_submodules
from utils.data_utils import get_dataset_path
from utils.data_loader import SycophancyDataLoader
from utils.configs_utils import DTYPE, DEVICE, N_BATCHES, EFFECTS_DIR
from utils.sfc_utils import prepare_ablation_masks, run_evaluation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class IdentityDict(nn.Module):
    """
    Dummy dictionary for Dense baseline.
    Updated to accept output_features and arbitrary kwargs.
    """
    def __init__(self, d_model):
        super().__init__()
        self.dict_size = d_model
        self.d_model = d_model

    def forward(self, x, output_features=False, **kwargs):
        if output_features:
            return x, x
        return x

    def encode(self, x): 
        return x

    def decode(self, f): 
        return f


def patch_dense_submodule(submod):
    """
    Injects get_activation and set_activation to standard PyTorch modules 
    for sfc_utils compatibility. Handles tuple outputs for models like Gemma.
    """
    def get_act():
        out = submod.output
        return out[0] if isinstance(out, tuple) else out
        
    def set_act(new_val):
        out = submod.output
        if isinstance(out, tuple):
            submod.output = (new_val,) + out[1:]
        else:
            submod.output = new_val
            
    submod.get_activation = get_act
    submod.set_activation = set_act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma")
    parser.add_argument("--experiment_name", type=str, default="nlp")
    parser.add_argument("--method", type=str, choices=["sfc", "dense"], required=True)
    args = parser.parse_args()

    if args.method == "sfc":
        k_steps = [0, 10, 50, 100, 300, 500, 1000, 2000, 5000]
        json_prefix = "extracted_sfc_features"
    else:
        k_steps = [0, 100, 500, 1000, 5000, 10000, 20000, 50000]
        json_prefix = "extracted_dense_neurons"

    json_path = Path(EFFECTS_DIR) / f"{json_prefix}_{args.experiment_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing extracted features file: {json_path}")

    with open(json_path, 'r') as f:
        extracted_data = json.load(f)

    syc_promoters = [feat for feat in extracted_data["ranked_features"] if feat.get("type") == "sycophancy_promoter"]

    print(f"Loading Model: gemma-2-2b for {args.method.upper()} Global Evaluation...")
    model = LanguageModel("google/gemma-2-2b", device_map=DEVICE, dispatch=True, attn_implementation="eager", dtype=DTYPE)
    model.requires_grad_(False)

    total_layers = 26
    d_model = 2304

    print("Loading SAEs/Dictionaries globally (Weights only)...")
    if args.method == "sfc":
        all_submodules, all_dictionaries = load_saes_and_submodules(
            model, start_layer=0, thru_layer=total_layers - 1, include_embed=False, dtype=DTYPE, device=DEVICE
        )
    else:
        all_submodules = []
        all_dictionaries = {}
        for l in range(total_layers):
            res_submod = model.model.layers[l]
            res_submod.name = f"resid_{l}"
            patch_dense_submodule(res_submod)
            all_submodules.append(res_submod)
            all_dictionaries[res_submod] = IdentityDict(d_model).to(DEVICE).to(DTYPE)

            mlp_submod = model.model.layers[l].mlp
            mlp_submod.name = f"mlp_{l}"
            patch_dense_submodule(mlp_submod)
            all_submodules.append(mlp_submod)
            all_dictionaries[mlp_submod] = IdentityDict(d_model).to(DEVICE).to(DTYPE)

    # Move ALL dictionaries to CPU immediately to free VRAM for clean model bound computation
    for dict_module in all_dictionaries.values():
        dict_module.to("cpu")
    t.cuda.empty_cache()

    dataloader = SycophancyDataLoader(file_path=get_dataset_path(args.experiment_name))
    tracer_kwargs = dict(scan=False, validate=False)

    completeness_results = {}
    faithfulness_raw = {}

    print("\n--- Computing Clean Model Bound ---")
    with t.no_grad():
        clean_syc, _ = run_evaluation(model, dataloader, [], {}, {}, tracer_kwargs)
    clean_score = clean_syc / 100.0
    print(f"Clean Model Score: {clean_score:.2%}")

    for k in k_steps:
        print(f"\n[Evaluating Global Top {k} Nodes]")
        if k == 0:
            completeness_results[k] = clean_score
            continue

        current_top_k = syc_promoters[:k]
        top_k_map = defaultdict(set)
        for feat in current_top_k:
            top_k_map[feat["submodule"]].add(feat["index"])

        active_submodules = [s for s in all_submodules if s.name in top_k_map]
        active_dictionaries = {s: all_dictionaries[s] for s in active_submodules}
        
        # Move ONLY the active dictionaries for this step back to GPU
        for s in active_submodules:
            active_dictionaries[s].to(DEVICE)
            
        t.cuda.empty_cache()

        # Metric 1: Completeness
        feats_to_ablate_comp = {s.name: list(top_k_map[s.name]) for s in active_submodules}
        comp_masks = prepare_ablation_masks(feats_to_ablate_comp, active_submodules, active_dictionaries, DEVICE)
        
        with t.no_grad():
            _, comp_syc = run_evaluation(model, dataloader, active_submodules, active_dictionaries, comp_masks, tracer_kwargs)
        completeness_results[k] = comp_syc / 100.0
        print(f"  -> Completeness (Ablate {k}): {completeness_results[k]:.2%}")

        # Metric 2: Faithfulness
        feats_to_ablate_faith = {}
        for s in active_submodules:
            dict_size = active_dictionaries[s].dict_size
            all_indices = set(range(dict_size))
            keep_indices = top_k_map[s.name]
            feats_to_ablate_faith[s.name] = list(all_indices - keep_indices)

        faith_masks = prepare_ablation_masks(feats_to_ablate_faith, active_submodules, active_dictionaries, DEVICE)
        
        with t.no_grad():
            _, faith_syc = run_evaluation(model, dataloader, active_submodules, active_dictionaries, faith_masks, tracer_kwargs)
        faithfulness_raw[k] = faith_syc / 100.0
        print(f"  -> Faithfulness (Keep {k}, Ablate rest in active layers): {faithfulness_raw[k]:.2%}")

        # Move active dictionaries back to CPU to prevent VRAM accumulation
        for s in active_submodules:
            active_dictionaries[s].to("cpu")

        # Aggressive memory clearing
        del comp_masks, faith_masks
        gc.collect()
        t.cuda.empty_cache()

    comp_output = Path(EFFECTS_DIR) / f"global_completeness_{args.method}_{args.experiment_name}.json"
    with open(comp_output, 'w') as f:
        json.dump({"method": args.method.upper(), "experiment": args.experiment_name, "data": completeness_results}, f, indent=4)

    faith_output = Path(EFFECTS_DIR) / f"global_faithfulness_{args.method}_{args.experiment_name}.json"
    with open(faith_output, 'w') as f:
        json.dump({
            "method": args.method.upper(),
            "experiment": args.experiment_name,
            "data": faithfulness_raw,
            "metadata": {"clean_score": clean_score}
        }, f, indent=4)

    print(f"\nAll Global Metrics successfully saved to {EFFECTS_DIR}!")


if __name__ == "__main__":
    main()