import os
import gc
import hashlib
import torch as t
from tqdm import tqdm
from nnsight import LanguageModel
from attribution import patching_effect
from dictionary_loading_utils import load_saes_and_submodules
from utils.data_loader import SycophancyDataLoader
from utils.configs_utils import (
    DTYPE, DEVICE, BATCH_SIZE, N_BATCHES,
    MODEL_NAME, EFFECTS_DIR, TOP_K_TO_ABLATE
)

from utils.sfc_utils import (
    print_top_features_and_links,
    generate_ablation_blacklist,
    prepare_ablation_masks,
    run_evaluation
)


def metric_fn(model, fac_ids, syc_ids):
    """
    Computes the logit difference between sycophantic and factual tokens.
    """
    logits = model.lm_head.output
    last_token_logits = logits[:, -1, :]
    batch_indices = t.arange(last_token_logits.size(0), device=DEVICE)
    syc_logits = last_token_logits[batch_indices, syc_ids]
    fac_logits = last_token_logits[batch_indices, fac_ids]
    return syc_logits - fac_logits


def run_experiment():
    """
    Executes the main attribution patching experiment and final evaluation.
    """
    os.makedirs(EFFECTS_DIR, exist_ok=True)
    tracer_kwargs = dict(scan=False, validate=False)

    print("Loading Model...")
    model = LanguageModel(
        MODEL_NAME,
        device_map=DEVICE,
        dispatch=True,
        attn_implementation="eager",
        torch_dtype=DTYPE
    )

    print("Loading SAEs and Submodules...")
    submodules, dictionaries = load_saes_and_submodules(
        model,
        thru_layer=None,
        include_embed=False,
        dtype=DTYPE,
        device=DEVICE,
    )

    dataloader = SycophancyDataLoader()
    batches = dataloader.get_text_batches(split="train", batch_size=BATCH_SIZE)

    print("Calculating Indirect Effects...")
    for batch_idx, (clean, fac_ids, syc_ids) in tqdm(enumerate(batches), total=N_BATCHES):
        if batch_idx == N_BATCHES:
            break

        hash_input = clean + [s.name for s in submodules]
        hash_str = ''.join(hash_input)
        hash_digest = hashlib.md5(hash_str.encode()).hexdigest()
        cache_path = os.path.join(EFFECTS_DIR, f"{hash_digest}.pt")

        if os.path.exists(cache_path):
            continue

        effects, *_ = patching_effect(
            clean,
            None,
            model,
            submodules,
            dictionaries,
            metric_fn,
            metric_kwargs=dict(fac_ids=fac_ids, syc_ids=syc_ids),
            method='ig'
        )

        to_save = {k.name: v.detach().to("cpu") for k, v in effects.items()}
        t.save(to_save, cache_path)

        del effects, _
        gc.collect()

    print("Aggregating Effects...")
    aggregated_effects = {submodule.name: 0 for submodule in submodules}

    for idx, (clean, fac_ids, syc_ids) in enumerate(batches):
        if idx == N_BATCHES:
            break

        hash_input = clean + [s.name for s in submodules]
        hash_str = ''.join(hash_input)
        hash_digest = hashlib.md5(hash_str.encode()).hexdigest()

        effects = t.load(os.path.join(EFFECTS_DIR, f"{hash_digest}.pt"))

        for submodule in submodules:
            aggregated_effects[submodule.name] += (
                effects[submodule.name].act[:, 1:, :]
            ).sum(dim=1).sum(dim=0)

    total_examples = BATCH_SIZE * N_BATCHES
    aggregated_effects = {k: v / total_examples for k,
                          v in aggregated_effects.items()}

    print("Extracting Top Features...")
    top_features = []
    for submod_name, effects in aggregated_effects.items():
        top_vals, top_idxs = t.topk(effects, k=5)
        for val, idx in zip(top_vals, top_idxs):
            if val > 0:
                top_features.append((val.item(), submod_name, idx.item()))

    top_features.sort(key=lambda x: x[0], reverse=True)

    print_top_features_and_links(top_features)

    feats_to_ablate = generate_ablation_blacklist(
        top_features, TOP_K_TO_ABLATE)

    print("Preparing ablation masks...")
    ablation_masks = prepare_ablation_masks(
        feats_to_ablate, submodules, dictionaries, DEVICE)

    run_evaluation(model, dataloader, submodules,
                   dictionaries, ablation_masks, tracer_kwargs)


if __name__ == "__main__":
    run_experiment()
