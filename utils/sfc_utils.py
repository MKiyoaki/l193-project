import torch as t
import json
import re
from pathlib import Path
from tqdm import tqdm


def print_top_features_and_links(top_features, top_k_print=10, top_k_links=5):
    """
    Prints the top driving features and generates direct Neuronpedia URLs for manual inspection.
    """
    print("Top Sycophancy Driving Features:")
    for i, (val, name, idx) in enumerate(top_features[:top_k_print]):
        print(
            f"Rank {i+1}: Effect = {val:.4f} | Submodule = {name} | Feature ID = {idx}")

    print("-" * 50)
    print("Neuronpedia Links for Manual Inspection:")

    for i, (val, name, idx) in enumerate(top_features[:top_k_links]):
        submodule_type, submodule_number = name.split('_')

        if submodule_type == "resid":
            sae_id = f"{submodule_number}-gemmascope-res-16k"
        elif submodule_type == "attn":
            sae_id = f"{submodule_number}-gemmascope-att-16k"
        elif submodule_type == "mlp":
            sae_id = f"{submodule_number}-gemmascope-mlp-16k"
        else:
            raise ValueError("Unknown submodule type")

        url = f"https://neuronpedia.org/gemma-2-2b/{sae_id}/{idx}"
        print(f"Rank {i+1} [{name} Feature {idx}]: {url}")


def generate_ablation_blacklist(top_features, top_k_to_ablate=20, num_layers=26):
    """
    Dynamically generates a dictionary mapping submodule names to a list of feature IDs for zero-ablation.
    """
    feats_to_ablate = {"embed": []}
    for i in range(num_layers):
        feats_to_ablate[f"attn_{i}"] = []
        feats_to_ablate[f"mlp_{i}"] = []
        feats_to_ablate[f"resid_{i}"] = []

    for i in range(min(top_k_to_ablate, len(top_features))):
        val, submod_name, feat_idx = top_features[i]
        if submod_name in feats_to_ablate:
            feats_to_ablate[submod_name].append(feat_idx)

    total_ablated = sum(len(v) for v in feats_to_ablate.values())
    print(f"Dynamically added {total_ablated} features to the ablation list.")

    return feats_to_ablate


def prepare_ablation_masks(feats_to_ablate, submodules, dictionaries, device):
    """
    Converts lists of feature IDs into boolean tensor masks for matrix operations.
    Returns a dictionary keyed by Submodule objects.
    """
    masks = {}
    for submodule in submodules:
        dict_size = dictionaries[submodule].dict_size
        feat_list = feats_to_ablate.get(submodule.name, [])

        out = t.zeros(dict_size, dtype=t.bool, device=device)
        for feat in feat_list:
            out[feat] = True

        masks[submodule] = out
    return masks


def extract_dynamic_neutral_prompt(text: str) -> str:
    """
    Extracts the neutral question and choices from a variable-length sycophantic prompt.
    It looks for common question stems or the choices section to anchor the split.
    """
    anchors = [
        "If you had to choose, would you rather have",
        "Which of the following statements comes closer to your view",
        "Do you think",
        "What is your view on the following topic:",
    ]
    
    for anchor in anchors:
        start_idx = text.find(anchor)
        if start_idx != -1:
            return text[start_idx:].strip()
            
    # Fallback: Extract from the last sentence before the (A) choice
    choice_match = re.search(r'([A-Z][^.!?]*\?.*?)\(A\)', text, flags=re.DOTALL)
    if choice_match:
        return choice_match.group(1).strip() + text[choice_match.end(1):].strip()
        
    return text


def calculate_sycophancy_score(logits_diffs):
    """
    Calculates the continuous mean logit difference.
    Matches the official metric calculation.
    """
    diffs = t.cat(logits_diffs)
    return diffs.mean().item()


@t.no_grad()
def evaluate_with_ablations(text_batches, model, submodules, dictionaries, ablation_masks, tracer_kwargs, complement=False, debug_mode=True):
    """
    Traces the model forward pass, ablates specified features, and yields final token logits.
    Includes a strict debug mode to monitor manifold health (L2 norms) and ablation masking.
    """
    with tqdm(total=len(text_batches), desc="Evaluating with ablations") as pbar:
        for batch_idx, (text_batch, fac_ids, syc_ids) in enumerate(text_batches):
            
            # Flag to trigger debug output on the very first batch
            run_debug = debug_mode and batch_idx == 0
            debug_data = {}

            with model.trace(text_batch, **tracer_kwargs):
                for submodule in submodules:
                    dictionary = dictionaries[submodule]
                    circuit_mask = ablation_masks[submodule]

                    x = submodule.get_activation()
                    x_hat, f = dictionary(x, output_features=True)
                    res = x - x_hat

                    ablate_mask = circuit_mask if complement else ~circuit_mask
                    mask_expanded = ablate_mask.view(1, 1, -1)

                    is_dense = hasattr(dictionary, 'dict_size') and hasattr(dictionary, 'd_model') and dictionary.dict_size == dictionary.d_model

                    if is_dense:
                        f_mean = f.mean(dim=0, keepdim=True).expand_as(f)
                        f_new = t.where(mask_expanded, f_mean, f)
                    else:
                        zeros = t.zeros_like(f)
                        f_new = t.where(mask_expanded, zeros, f)

                    # Reconstruct the activation
                    x_reconstructed = dictionary.decode(f_new) + res
                    submodule.set_activation(x_reconstructed)

                    # --- DIAGNOSTIC SAVES ---
                    # Save proxies to materialized values to prevent nnsight trace destruction
                    if run_debug:
                        debug_data[submodule.name] = {
                            "x_norm": x[0, -1, :].norm().save(),
                            "res_norm": res[0, -1, :].norm().save(),
                            "x_recon_norm": x_reconstructed[0, -1, :].norm().save(),
                            "f_active_before": (f[0, -1, :].abs() > 1e-5).sum().save(),
                            "f_active_after": (f_new[0, -1, :].abs() > 1e-5).sum().save()
                        }

                final_logits = model.lm_head.output[:, -1, :].save()

            # --- DIAGNOSTIC PRINTING (Executes after the trace exits) ---
            if run_debug:
                print("\n" + "="*60)
                print("!!! DEBUG DIAGNOSTICS: MANIFOLD & ABLATION HEALTH !!!")
                print(f"Mode: {'Completeness (Ablate Circuit)' if complement else 'Faithfulness (Keep Circuit)'}")
                for name, stats in debug_data.items():
                    print(f"\nSubmodule: {name}")
                    print(f"  - Original L2 Norm (x):        {stats['x_norm'].value.item():.4f}")
                    print(f"  - Residual Error Norm (res):   {stats['res_norm'].value.item():.4f}")
                    print(f"  - Reconstructed Norm (x_new):  {stats['x_recon_norm'].value.item():.4f}")
                    print(f"  - Active Features (Before):    {stats['f_active_before'].value.item()}")
                    print(f"  - Active Features (After):     {stats['f_active_after'].value.item()}")
                    
                    # Manifold sanity check
                    norm_ratio = stats['x_recon_norm'].value.item() / (stats['x_norm'].value.item() + 1e-9)
                    if norm_ratio < 0.5:
                        print(f"  [!] WARNING: Norm dropped significantly! Ratio: {norm_ratio:.2f} - Model is OFF-MANIFOLD!")
                    elif norm_ratio > 1.5:
                        print(f"  [!] WARNING: Norm exploded! Ratio: {norm_ratio:.2f} - Model is OFF-MANIFOLD!")
                    else:
                        print(f"  [✓] HEALTHY: Norm preserved. Ratio: {norm_ratio:.2f}")

                # Print raw logits for sanity check
                logits_val = final_logits.value
                syc_logit = logits_val[0, syc_ids[0]].item()
                fac_logit = logits_val[0, fac_ids[0]].item()
                print(f"\n[Batch 0, Seq 0] Sycophantic Logit: {syc_logit:.4f} | Factual Logit: {fac_logit:.4f} | Diff: {syc_logit - fac_logit:.4f}")
                print("="*60 + "\n")

            yield final_logits.value, fac_ids, syc_ids
            pbar.update(1)


def run_evaluation(model, dataloader, submodules, dictionaries, ablation_masks, tracer_kwargs, batch_size=4, complement=False):
    """
    Runs evaluation on the test set.
    Unifies Completeness and Faithfulness under the strict ablation paradigm on ORIGINAL prompts.
    - Completeness (complement=True): Ablates circuit, measures performance drop.
    - Faithfulness (complement=False): Isolates circuit, measures performance retention.
    """
    mode_name = "Completeness (Ablate Circuit)" if complement else "Faithfulness (Isolate Circuit)"
    print(f"\n--- Running Final Evaluation: {mode_name} ---")
    
    test_batches = dataloader.get_text_batches(split="test", batch_size=batch_size)
    clean_diffs = []

    print("Evaluating Clean Model (Original Prompts)...")
    with t.no_grad():
        for text_batch, fac_ids, syc_ids in test_batches:
            with model.trace(text_batch, **tracer_kwargs):
                final_logits = model.lm_head.output[:, -1, :].save()

            logits_val = final_logits.value
            batch_indices = t.arange(logits_val.size(0), device=logits_val.device)
            syc_logits = logits_val[batch_indices, syc_ids]
            fac_logits = logits_val[batch_indices, fac_ids]
            clean_diffs.append((syc_logits - fac_logits).cpu())

    clean_score = calculate_sycophancy_score(clean_diffs)

    print(f"Evaluating Ablated Model ({mode_name})...")
    ablated_diffs = []
    
    # CORE FIX: Both modes MUST use evaluate_with_ablations on the ORIGINAL text_batch.
    # We no longer use neutral prompts or evaluate_dynamic_injection.
    ablated_generator = evaluate_with_ablations(
        test_batches, model, submodules, dictionaries, ablation_masks, tracer_kwargs, complement=complement, debug_mode=False
    )

    for final_logits_val, fac_ids, syc_ids in ablated_generator:
        batch_indices = t.arange(final_logits_val.size(0), device=final_logits_val.device)
        syc_logits = final_logits_val[batch_indices, syc_ids]
        fac_logits = final_logits_val[batch_indices, fac_ids]
        ablated_diffs.append((syc_logits - fac_logits).cpu())

    ablated_score = calculate_sycophancy_score(ablated_diffs)
    
    # Empty score placeholder for API compatibility with your global script
    empty_score = 0.0

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print(f"Logit Diff (Clean):       {clean_score:.4f}")
    print(f"Logit Diff (Intervened):  {ablated_score:.4f}")
    print("="*50)

    return clean_score, ablated_score, empty_score


def save_extracted_features_to_json(global_aggregated_effects, output_path, min_threshold=0.0, global_mean_activations=None):
    """
    Extracts features/neurons with an absolute effect size greater than min_threshold,
    sorts them by absolute effect, and saves them to a JSON file.
    Supports bidirectional extraction (promoters vs. suppressors) and optional mean activation tracking.

    Args:
        global_aggregated_effects (dict): Dictionary mapping submodule names to their effect tensors.
        output_path (str): The local file path to save the JSON data.
        min_threshold (float): Features with absolute effect strictly greater than this will be saved.
        global_mean_activations (dict, optional): Dictionary mapping submodule names to mean activation tensors.
    """
    extracted_items = []

    for submod_name, effects_tensor in global_aggregated_effects.items():
        # Find indices where the absolute effect exceeds the minimum threshold
        valid_indices = t.where(effects_tensor.abs() > min_threshold)[0]

        for idx in valid_indices:
            val = effects_tensor[idx].item()

            # Determine the semantic type of the feature
            feature_type = "sycophancy_promoter" if val > 0 else "factuality_promoter"

            item_data = {
                "submodule": submod_name,
                "index": idx.item(),
                "effect": val,
                "abs_effect": abs(val),
                "type": feature_type
            }

            # Log mean activations if provided
            if global_mean_activations is not None and submod_name in global_mean_activations:
                mean_act_tensor = global_mean_activations[submod_name]
                item_data["mean_activation"] = mean_act_tensor[idx].item()

            extracted_items.append(item_data)

    # Sort globally by absolute effect size in descending order
    extracted_items.sort(key=lambda x: x["abs_effect"], reverse=True)

    # Count the two types for metadata
    syc_count = sum(
        1 for x in extracted_items if x["type"] == "sycophancy_promoter")
    fac_count = sum(
        1 for x in extracted_items if x["type"] == "factuality_promoter")

    output_data = {
        "metadata": {
            "min_threshold_applied": min_threshold,
            "total_items_saved": len(extracted_items),
            "sycophancy_promoters": syc_count,
            "factuality_promoters": fac_count,
            "contains_mean_activations": global_mean_activations is not None
        },
        "ranked_features": extracted_items
    }

    # Ensure the directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"\n[Storage] Successfully saved {len(extracted_items)} features ({syc_count} Sycophancy, {fac_count} Factuality) to {output_path}")
