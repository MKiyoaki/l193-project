# What Does Sparse Feature Circuits Capture? A Study of Feature Circuits in Sycophantic Behaviour

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Model Gemma--2--2B](https://img.shields.io/badge/Model-Gemma--2--2B-orange.svg)](https://huggingface.co/google/gemma-2-2b)
[![SAE GemmaScope](https://img.shields.io/badge/SAE-GemmaScope--16k-purple.svg)](https://huggingface.co/google/gemma-scope-2b-pt-res)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Cambridge L193 Project](https://img.shields.io/badge/University%20of%20Cambridge-L193%20XAI-cyan.svg)](https://www.cl.cam.ac.uk/)

> **Author**: Yifei Shi (`ys690@cam.ac.uk`) — Department of Computer Science and Technology, University of Cambridge  
> **Course / Report**: L193 Explainable AI Project Report  
> **Supervisors & Organizers**: Prof. Mateja Jamnik, Mateo Espinosa Zarlenga, Dr. Zohreh Shams

---

## 📌 Executive Summary

Large Language Models (LLMs) frequently exhibit **sycophancy**—a complex alignment failure where models preferentially conform to user-provided premises at the expense of factual accuracy. While **Sparse Feature Circuits (SFC)** and the **SHIFT framework** ([Marks et al., 2024](https://arxiv.org/abs/2403.19647)) have shown notable success in constrained linguistic tasks (e.g., subject-verb agreement on small models), their ability to discover interpretable mechanisms in complex social alignment tasks remains underexplored.

This repository contains the official codebase and data for investigating sparse causal feature subgraphs of sycophantic behaviour in **Gemma-2-2B** using **GemmaScope 16k JumpReLU Sparse Autoencoders (SAEs)** across three distinct domains:
1. **Natural Language Processing (NLP) Survey**: Academic controversies with objective factual baselines.
2. **Philosophy (PHI)**: Abstract philosophical & ideological stances.
3. **Political Typology (POLITICAL)**: Socio-economic value matching.

<p align="center">
  <img src="docs/imgs/example_syco.png" alt="Sycophancy Example of Die Brücke" width="85%"/>
  <br/>
  <em>Figure 1: Demonstration of sycophantic behaviour on Gemma-2-2B. When prompted neutrally, the model correctly identifies Expressionism; when introduced to a user's biased premise, the model adopts the incorrect Impressionist label.</em>
</p>

---

## 🔑 Key Research Findings

1. **Substantial Behavioural Mitigation via SHIFT**: Intervening on circuits containing only $10^2$ to $10^3$ SAE features reduces the Sycophancy Rate from **85.23% to 45.22%** in the NLP domain and **68.39% to 35.06%** in Philosophy, while preserving general reasoning abilities (MMLU normal accuracy ~37% vs ~30% for random baselines).
2. **Dominance of Surface-Level Formatting Shortcuts**: Discovered causal nodes primarily capture high-impact structural and positional cues (e.g., option letter markers like `"A"` / `"B"`, section start tokens, colon delimiters) rather than deep semantic reasoning.
3. **Intervention Overshoot & Non-Linear Dynamics**: Isolating sycophancy promoters while zero-ablating distributed factuality guardrails causes a dramatic surge in logit difference past unablated baseline levels, exposing competing subnetwork dynamics.
4. **Domain Instability & Causal Dilution**: Features exhibit significant functional shifts across domains (e.g., feature `resid_17 / 7127` promotes sycophancy in NLP but suppresses it in Philosophy). Furthermore, highly polarized political topics display weak, dispersed feature activation with minimal localized shortcut dominance.

---

## 📊 Experimental Results & Visualizations

### 1. Circuit Completeness and Faithfulness Trajectories

We compare **Sparse Feature Circuits (SFC)** against a **Dense Neurons Baseline** and a **Random Feature Baseline** across all three sycophancy domains. Completeness measures circuit necessity (driving logit difference to the empty model baseline), while Faithfulness measures circuit sufficiency.

| NLP Survey Subset | Philosophy Subset | Political Typology Subset |
| :---: | :---: | :---: |
| ![Ablation NLP](docs/imgs/abl_nlp.png) | ![Ablation Philosophy](docs/imgs/abl_phi.png) | ![Ablation Political](docs/imgs/abl_political.png) |

* **Key Takeaway**: Across all domains, SFC (blue) efficiently drives Completeness down using $10^2 - 10^3$ nodes. In contrast, the Dense Baseline (orange) fails catastrophically due to dense polysemanticity and out-of-distribution state collapses.

---

### 2. Representative Driving Features (NLP Survey Domain)

Below are top-ranked SAE features in Gemma-2-2B driving sycophantic outcomes, categorized by their indirect causal effect ($IE$):

| Feature ID | Role | Mean $IE$ | Neuronpedia Semantic Label | Circuit Freq | Interactive Link |
| :--- | :---: | :---: | :--- | :---: | :---: |
| <mark style="background-color: #fee8e8;">`resid_17 / 7127`</mark> | **Promoter** | $+2.3750$ | Beginning of new sections/topics in text | 10/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/17-gemmascope-res-16k/7127) |
| <mark style="background-color: #fee8e8;">`resid_18 / 3851`</mark> | **Promoter** | $+1.6094$ | Follows pronouns | 10/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/18-gemmascope-res-16k/3851) |
| <mark style="background-color: #fee8e8;">`mlp_25 / 6001`</mark> | **Promoter** | $+1.4609$ | Questions and statement formats | 10/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/25-gemmascope-mlp-16k/6001) |
| <mark style="background-color: #fee8e8;">`resid_20 / 6631`</mark> | **Promoter** | $+1.4219$ | Beginning of text / important markers | 10/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/6631) |
| <mark style="background-color: #fee8e8;">`resid_24 / 11396`</mark> | **Promoter** | $+1.2891$ | Letter "B"; Technical terms | 10/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/24-gemmascope-res-16k/11396) |
| <mark style="background-color: #e8f0fe;">`resid_24 / 6976`</mark> | **Suppressor** | $-3.0156$ | Q&A format; Sequence begin with "A" | 11/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/24-gemmascope-res-16k/6976) |
| <mark style="background-color: #e8f0fe;">`resid_25 / 6574`</mark> | **Suppressor** | $-2.9219$ | Offers and negotiations terms | 10/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/25-gemmascope-res-16k/6574) |
| <mark style="background-color: #e8f0fe;">`resid_23 / 9126`</mark> | **Suppressor** | $-1.8984$ | Numeric data & statistical figures | 10/11 | [Neuronpedia](https://neuronpedia.org/gemma-2-2b/23-gemmascope-res-16k/9126) |

---

### 3. Feature Activations, Topography & Stability

<p align="center">
  <img src="docs/imgs/activation_nlp.png" alt="Token Level Activation Map" width="95%"/>
  <br/>
  <em>Figure 2: Token-level activation maps for top sycophancy promoter vs suppressor. Suppressors activate on localized formatting markers (option letters), whereas dominant promoters function as dense background features with high activation on <code>&lt;bos&gt;</code> tokens.</em>
</p>

<p align="center">
  <img src="docs/imgs/circuit_nlp.png" alt="Causal Subgraph Topography" width="85%"/>
  <br/>
  <em>Figure 3: Topology of a 15-node causal subgraph driving sycophancy in NLP Survey. Information flows from early structural tracking features (Layers 8-17) to actuator nodes (<code>mlp_25 / 6001</code>) regulating response formats.</em>
</p>

<p align="center">
  <img src="docs/imgs/feature_heatmap.png" alt="Cross-Domain Feature Heatmap" width="80%"/>
  <br/>
  <em>Figure 4: Cross-domain stability heatmap of top features across NLP, Philosophy, and Political Typology, revealing functional shifts and causal dilution.</em>
</p>

---

### 4. Behavioural Mitigation Results (SHIFT Intervention)

| Domain | Model Condition | Sycophancy Rate (%) ↓ | Normal Accuracy / MMLU (%) ↑ |
| :--- | :--- | :---: | :---: |
| **NLP Survey** | Clean Model | **85.23** | **48.13** |
| | **SHIFT Intervened** | **45.22** | **36.92** |
| | Random Feature Baseline | 43.62 | 30.72 |
| **Philosophy** | Clean Model | **68.39** | **48.13** |
| | **SHIFT Intervened** | **35.06** | **37.38** |
| | Random Feature Baseline | 41.03 | 30.76 |
| **Political Typology** | Clean Model | **50.29** | **48.13** |
| | **SHIFT Intervened** | **41.86** | **37.49** |
| | Random Feature Baseline | 43.63 | 30.40 |

---

## 🛠️ Repository Architecture

The repository is organized into active pipeline code for our Gemma-2-2B sycophancy paper and legacy baseline modules:

```
l193-project/
├── data/                           # Active Anthropic Sycophancy sub-datasets
│   ├── sycophancy_nlp.jsonl
│   ├── sycophancy_phi.jsonl
│   └── sycophancy_political.jsonl
├── experiments/                    # Main execution & evaluation scripts
│   ├── run_bib_shift.py            # SFC feature extraction & threshold completeness
│   ├── run_dense_baseline.py       # Dense neuron baseline extraction & eval
│   ├── eval_global.py              # Global faithfulness & completeness evaluation
│   ├── eval_completeness.py        # Completeness curve plotting
│   └── eval_faithfulness.py        # Faithfulness curve plotting
├── scripts/                        # Automated shell scripts for experiments
│   ├── process_sycophancy_*.sh     # Data download & formatting scripts
│   ├── run_bib_shift_gemma_*.sh    # SFC feature extraction execution
│   ├── run_dense_gemma_*.sh        # Dense baseline execution
│   └── eval_*.sh                   # Evaluation & plotting execution
├── tools/                          # Helper tools & Neuronpedia link generators
│   ├── process_dataset.py          # Anthropic dataset formatter
│   └── generate_neuropedia.py      # Neuronpedia URL generator for extracted SAE features
├── utils/                          # Pipeline data loaders & SFC helpers
│   ├── data_loader.py              # SycophancyDataLoader for Anthropic JSONL
│   ├── sfc_utils.py                # Masking, ablation, and evaluation helpers
│   ├── configs_utils.py            # Hardware & model execution configurations
│   └── data_processor/             # Raw HuggingFace dataset handler
├── effects/                        # JSON results, extracted feature sets, and plots
├── manuscript/                     # Paper report source (LaTeX, BibTeX, & figures)
│   └── L193_Report/
├── docs/imgs/                      # Embedded paper figures for README & docs
├── attribution.py                  # Integrated Gradients & patching attribution logic
├── dictionary_loading_utils.py     # GemmaScope & Pythia SAE loading utilities
├── activation_utils.py             # SparseAct wrapper for activations & error terms
├── ablation.py                     # SAE feature zero/mean ablation primitives
├── loading_utils.py                # Submodule definitions & model hooks
├── requirements.txt                # Dependencies
└── legacy/                         # Upstream baseline code (Pythia-70M, agreement tasks)
    ├── code/                       # Upstream circuit discovery (circuit.py, coo_utils.py)
    ├── data/                       # Subject-verb agreement datasets (rc, nounpp, simple)
    ├── annotations/                # Pythia 70M feature annotations
    ├── experiments/                # Original benchmark Jupyter notebooks
    └── scripts/                    # Upstream shell scripts
```

---

## ⚡ Installation & Setup

### Prerequisites
* Python $\ge 3.10$
* PyTorch $\ge 2.0$ with CUDA support
* NVIDIA GPU with $\ge 24\text{GB}$ VRAM (recommended for Gemma-2-2B SAE execution)

### 1. Environment Setup

```bash
git clone https://github.com/MKiyoaki/l193-project.git
cd l193-project

python -m venv venv
source venv/bin/activate  # Or: conda create -n sfc python=3.10 && conda activate sfc

pip install -r requirements.txt
```

### 2. Hugging Face Access & GemmaScope SAEs
Ensure you have logged into Hugging Face to access `google/gemma-2-2b` and `google/gemma-scope-2b-pt-*`:
```bash
huggingface-cli login
```

---

## 🚀 Execution & Reproduction Guide

### Step 1: Preprocess Datasets
Download and format the Anthropic Sycophancy sub-datasets:
```bash
# Process NLP Survey, Philosophy, and Political Typology subsets
bash scripts/process_sycophancy_nlp.sh
bash scripts/process_sycophancy_phi.sh
bash scripts/process_sycophancy_political.sh
```

---

### Step 2: Extract SFC Features & Evaluate Completeness
Extract sparse feature circuits via Integrated Gradients (IG) and evaluate threshold completeness on Gemma-2-2B:

```bash
# Run SFC extraction on NLP Survey domain
bash scripts/run_bib_shift_gemma_nlp.sh

# Run SFC extraction on Philosophy domain
bash scripts/run_bib_shift_gemma_phi.sh

# Run SFC extraction on Political Typology domain
bash scripts/run_bib_shift_gemma_political.sh
```

*Outputs will be saved under `effects/extracted_sfc_features_<domain>.json` and `effects/sfc_completeness_results_<domain>.json`.*

---

### Step 3: Run Dense Neurons Baseline
Extract dense neuron attributions to benchmark against sparse feature circuits:

```bash
bash scripts/run_dense_gemma_nlp.sh
bash scripts/run_dense_gemma_phi.sh
bash scripts/run_dense_gemma_political.sh
```

---

### Step 4: Evaluate Global Faithfulness & Completeness Trajectories
Run global threshold evaluations across circuit sizes:

```bash
# Evaluate SFC global completeness and faithfulness
bash scripts/eval_global_sfc_nlp.sh
bash scripts/eval_global_sfc_phi.sh
bash scripts/eval_global_sfc_political.sh
```

---

### Step 5: Generate Plots and Interactive Links
Generate normalized completeness/faithfulness curve plots and Neuronpedia links:

```bash
# Plot completeness curves for NLP subset
python experiments/eval_completeness.py --experiment_name nlp

# Generate Neuronpedia links for top extracted features
python tools/generate_neuropedia.py --file effects/extracted_sfc_features_nlp.json --top_k 10
```

---

## 🔗 Neuronpedia Integration

You can easily inspect any extracted GemmaScope feature on [Neuronpedia](https://www.neuronpedia.org/). The tool formats URLs automatically:
```bash
python tools/generate_neuropedia.py --file effects/extracted_sfc_features_nlp.json --top_k 5
```
Example Output:
```text
Loaded 1240 total features from effects/extracted_sfc_features_nlp.json

### Top 5 Sycophancy Promoters
------------------------------------------------------------
1. **[resid_17 / Feature 7127](https://neuronpedia.org/gemma-2-2b/17-gemmascope-res-16k/7127)** | Effect: `2.3750`
2. **[resid_18 / Feature 3851](https://neuronpedia.org/gemma-2-2b/18-gemmascope-res-16k/3851)** | Effect: `1.6094`
3. **[mlp_25 / Feature 6001](https://neuronpedia.org/gemma-2-2b/25-gemmascope-mlp-16k/6001)** | Effect: `1.4609`
4. **[resid_20 / Feature 6631](https://neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/6631)** | Effect: `1.4219`
5. **[resid_24 / Feature 11396](https://neuronpedia.org/gemma-2-2b/24-gemmascope-res-16k/11396)** | Effect: `1.2891`
```

---

## 📚 Citations & Acknowledgments

If you find this codebase or report useful in your research, please cite our report as well as the foundational Sparse Feature Circuits paper:

```bibtex
@article{shi2026sycophancy_sfc,
  title={What Does Sparse Feature Circuits Capture? A Study of Feature Circuits in Sycophantic Behaviour},
  author={Yifei Shi},
  journal={L193 Explainable AI Project Report, University of Cambridge},
  year={2026}
}

@inproceedings{marks2025sparse,
  title={Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models},
  author={Samuel Marks and Can Rager and Eric J Michaud and Yonatan Belinkov and David Bau and Aaron Mueller},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=I4e82CIDxv}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
