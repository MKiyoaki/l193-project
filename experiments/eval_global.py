# experiments/eval_global.py

import os
import gc
import json
import argparse
import torch as t
from torch import nn
from nnsight import LanguageModel
from pathlib import Path

from dictionary_loading_utils import load_saes_and_submodules
from utils.data_utils import get_dataset_path
from utils.data_loader import SycophancyDataLoader
from utils.configs_utils import DTYPE, DEVICE, EFFECTS_DIR
from utils.sfc_utils import prepare_ablation_masks, run_evaluation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class IdentityDict(nn.Module):
    """
    Dummy dictionary for Dense baseline.
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
    Injects get_activation and set_activation.
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
    parser.add_argument(
        "--thresholds", 
        type=float, 
        nargs="+", 
        default=[2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005],
        help="List of absolute effect thresholds for defining the circuit."
    )
    args = parser.parse_args()

    json_prefix = "extracted_sfc_features" if args.method == "sfc" else "extracted_dense_neurons"
    json_path = Path(EFFECTS_DIR) / f"{json_prefix}_{args.experiment_name}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Missing extracted features file: {json_path}")

    with open(json_path, 'r') as f:
        extracted_data = json.load(f)

    print(f"Loading Model: {args.model} for {args.method.upper()} Global Evaluation...")
    model_id = "google/gemma-2-2b" if "gemma" in args.model.lower() else "EleutherAI/pythia-70m-deduped"
    model = LanguageModel(model_id, device_map=DEVICE, dispatch=True, attn_implementation="eager", dtype=DTYPE)
    model.requires_grad_(False)

    total_layers = 26 if "gemma" in args.model.lower() else 6
    d_model = 2304 if "gemma" in args.model.lower() else 512

    print("Loading SAEs/Dictionaries globally (Weights only)...")
    if args.method == "sfc":
        all_submodules, all_dictionaries = load_saes_and_submodules(
            model, start_layer=0, thru_layer=total_layers - 1, include_embed=False, dtype=DTYPE, device=DEVICE
        )
    else:
        all_submodules = []
        all_dictionaries = {}
        for l in range(total_layers):
            res_submod = model.model.layers[l] if hasattr(model.model, 'layers') else model.gpt_neox.layers[l]
            res_submod.name = f"resid_{l}"
            patch_dense_submodule(res_submod)
            all_submodules.append(res_submod)
            all_dictionaries[res_submod] = IdentityDict(d_model).to(DEVICE).to(DTYPE)

            mlp_submod = res_submod.mlp
            mlp_submod.name = f"mlp_{l}"
            patch_dense_submodule(mlp_submod)
            all_submodules.append(mlp_submod)
            all_dictionaries[mlp_submod] = IdentityDict(d_model).to(DEVICE).to(DTYPE)

    start_layer = 8 if "gemma" in args.model.lower() else 2
    all_submodules = [s for s in all_submodules if int(s.name.split('_')[-1]) >= start_layer]
    valid_submod_names = set([s.name for s in all_submodules])

    circuit_nodes = [
        feat for feat in extracted_data["ranked_features"] 
        if feat["submodule"] in valid_submod_names
    ]

    dataloader = SycophancyDataLoader(file_path=get_dataset_path(args.experiment_name))
    tracer_kwargs = dict(scan=False, validate=False)

    print("\n--- Establishing Base Metrics ---")
    for s in all_submodules:
        all_dictionaries[s].to(DEVICE)
    t.cuda.empty_cache()

    empty_indices = {s.name: [] for s in all_submodules}
    empty_masks = prepare_ablation_masks(empty_indices, all_submodules, all_dictionaries, DEVICE)

    with t.no_grad():
        _, clean_score, _ = run_evaluation(
            model, dataloader, all_submodules, all_dictionaries, empty_masks, 
            tracer_kwargs, batch_size=4, complement=True
        )
    print(f"Clean Model Score: {clean_score:.4f}")

    with t.no_grad():
        _, empty_score, _ = run_evaluation(
            model, dataloader, all_submodules, all_dictionaries, empty_masks, 
            tracer_kwargs, batch_size=4, complement=False
        )
    print(f"Empty Model Score: {empty_score:.4f}")

    for s in all_submodules:
        all_dictionaries[s].to("cpu")
    del empty_masks
    gc.collect()
    t.cuda.empty_cache()

    completeness_results = {0: clean_score}
    faithfulness_raw = {0: empty_score}

    for threshold in sorted(args.thresholds, reverse=True):
        current_circuit = [feat for feat in circuit_nodes if feat["abs_effect"] >= threshold]
        n_nodes = len(current_circuit)
        
        print(f"\n[Evaluating Threshold {threshold} | Nodes in Circuit: {n_nodes}]")
        if n_nodes == 0:
            continue

        circuit_indices = {s.name: [] for s in all_submodules}
        for feat in current_circuit:
            circuit_indices[feat["submodule"]].append(feat["index"])

        for s in all_submodules:
            all_dictionaries[s].to(DEVICE)
        t.cuda.empty_cache()

        circuit_masks = prepare_ablation_masks(circuit_indices, all_submodules, all_dictionaries, DEVICE)
        
        with t.no_grad():
            _, comp_syc, _ = run_evaluation(
                model, dataloader, all_submodules, all_dictionaries, circuit_masks, 
                tracer_kwargs, batch_size=4, complement=True
            )
        completeness_results[n_nodes] = comp_syc
        print(f"  -> Completeness: {completeness_results[n_nodes]:.4f}")

        with t.no_grad():
            _, faith_syc, _ = run_evaluation(
                model, dataloader, all_submodules, all_dictionaries, circuit_masks, 
                tracer_kwargs, batch_size=4, complement=False
            )
        faithfulness_raw[n_nodes] = faith_syc
        print(f"  -> Faithfulness: {faithfulness_raw[n_nodes]:.4f}")

        for s in all_submodules:
            all_dictionaries[s].to("cpu")
        
        del circuit_masks
        gc.collect()
        t.cuda.empty_cache()

    comp_output = Path(EFFECTS_DIR) / f"global_completeness_{args.method}_{args.experiment_name}.json"
    with open(comp_output, 'w') as f:
        json.dump({
            "method": args.method.upper(), 
            "experiment": args.experiment_name, 
            "data": completeness_results,
            "metadata": {"clean_score": clean_score, "empty_score": empty_score}
        }, f, indent=4)

    faith_output = Path(EFFECTS_DIR) / f"global_faithfulness_{args.method}_{args.experiment_name}.json"
    with open(faith_output, 'w') as f:
        json.dump({
            "method": args.method.upper(),
            "experiment": args.experiment_name,
            "data": faithfulness_raw,
            "metadata": {"clean_score": clean_score, "empty_score": empty_score}
        }, f, indent=4)

    print(f"\nAll Global Metrics successfully saved to {EFFECTS_DIR}!")

if __name__ == "__main__":
    main()