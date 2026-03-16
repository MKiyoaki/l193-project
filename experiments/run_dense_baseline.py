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
        # Identity mapping: features are the same as activations
        return f


def metric_fn(model, fac_ids, syc_ids):
    logits = model.lm_head.output
    last_token_logits = logits[:, -1, :]
    batch_indices = t.arange(last_token_logits.size(0), device=DEVICE)
    syc_logits = last_token_logits[batch_indices, syc_ids]
    fac_logits = last_token_logits[batch_indices, fac_ids]
    return syc_logits - fac_logits


def get_submodules(model, start_layer, end_layer):
    """
    Retrieves the actual residual stream / MLP submodule objects for the dense model.
    """
    submodules = []
    for layer_idx in range(start_layer, end_layer + 1):
        res_submod = model.model.layers[layer_idx]
        res_submod.name = f"resid_{layer_idx}"
        submodules.append(res_submod)

        mlp_submod = model.model.layers[layer_idx].mlp
        mlp_submod.name = f"mlp_{layer_idx}"
        submodules.append(mlp_submod)

    return submodules


def evaluate_completeness_curve(
    model,
    dataloader,
    submodules,
    ranked_neurons,
    k_steps,
    d_model,
    device,
    tracer_kwargs
):
    """
    Evaluates sycophancy rate using mean ablation via out-of-place math.
    """
    # Materialize dataloader once to prevent iterator exhaustion
    batches = list(dataloader)

    curve_results = {}

    for k in k_steps:
        print(f"\nEvaluating mean ablation for Top {k} Neurons...")
        current_top_k = ranked_neurons[:k]

        ablation_masks = {}
        for submod in submodules:
            ablation_masks[submod.name] = t.zeros(
                d_model, dtype=t.float32, device=device
            )

        for _, submod_name, idx in current_top_k:
            if submod_name in ablation_masks:
                ablation_masks[submod_name][idx] = 1.0

        syc_logits_list = []
        fac_logits_list = []

        with tqdm(total=len(batches), desc=f"Top {k} Eval") as pbar:
            for text_batch, fac_ids, syc_ids in batches:

                # Defensively align index tensors to target device
                fac_ids_dev = fac_ids.to(device)
                syc_ids_dev = syc_ids.to(device)

                with model.trace(text_batch, **tracer_kwargs):

                    for submod in submodules:
                        # Extract proxy safely without strict string/tuple checks
                        # Gemma's residual layers return a tuple, MLPs return a tensor
                        output_proxy = submod.output

                        # Use .shape to infer if we need to index [0]
                        # Tensors have .shape, tuples don't
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

                # Extract values to CPU immediately after trace context
                syc_logits_list.append(s_proxy.value.detach().cpu())
                fac_logits_list.append(f_proxy.value.detach().cpu())

                pbar.update(1)

        all_syc = t.cat(syc_logits_list)
        all_fac = t.cat(fac_logits_list)

        syc_rate = (all_syc > all_fac).float().mean().item()
        curve_results[k] = syc_rate
        print(f"Result for Top {k}: Sycophancy Rate = {syc_rate:.4%}")

    return curve_results


def run_dense_experiment(args):
    os.makedirs(EFFECTS_DIR, exist_ok=True)
    tracer_kwargs = dict(scan=False, validate=False)

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

    dataloader = SycophancyDataLoader()
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

            for submodule in submodules:
                tensor_obj = raw_effects[submodule]

                # CRITICAL FIX: Extract the raw tensor from SparseAct wrapper
                if hasattr(tensor_obj, 'act'):
                    tensor_data = tensor_obj.act
                else:
                    tensor_data = tensor_obj

                # Now tensor_data is a torch.Tensor and has .ndim
                if tensor_data.ndim == 3:
                    chunk_aggregated_effects[submodule.name] += (
                        tensor_data[:, 1:, :]
                    ).sum(dim=1).sum(dim=0).to("cpu")
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
    global_aggregated_effects = {
        k: v / total_examples for k, v in global_aggregated_effects.items()
    }

    print("\nExtracting Global Top Dense Neurons...")
    top_neurons = []
    for submod_name, effects_tensor in global_aggregated_effects.items():
        top_vals, top_idxs = t.topk(effects_tensor, k=10)
        for val, idx in zip(top_vals, top_idxs):
            if val > 0:
                top_neurons.append((val.item(), submod_name, idx.item()))

    top_neurons.sort(key=lambda x: x[0], reverse=True)

    K_STEPS = [5, 10, 15, 20, 50]

    print("\n--- Starting Dense Neuron Completeness Evaluation ---")

    eval_batches = islice(dataloader.get_text_batches(
        split="test", batch_size=BATCH_SIZE), N_BATCHES)

    # Fetch ALL submodules across the network for full evaluation masking
    all_submodules = get_submodules(model, 0, total_layers - 1)

    curve_results = evaluate_completeness_curve(
        model=model,
        dataloader=eval_batches,
        submodules=all_submodules,
        ranked_neurons=top_neurons,
        k_steps=K_STEPS,
        d_model=d_model,
        device=DEVICE,
        tracer_kwargs=tracer_kwargs
    )

    print("\n--- Final Dense Completeness Curve Results ---")
    for k, rate in curve_results.items():
        print(f"Top {k} Neurons Ablated -> Sycophancy Rate: {rate:.4%}")

    # Save results to a local JSON file
    output_data = {
        "method": "Dense_Neurons",
        "data": curve_results
    }

    output_path = "dense_completeness_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"\nDense Completeness Curve results successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma")
    args = parser.parse_args()
    run_dense_experiment(args)


if __name__ == "__main__":
    main()
