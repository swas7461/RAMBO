# RAMBO: Reliability Analysis for Mamba through Bit-flip Attack Optimization

**RAMBO** is a Bit-Flip Attack (BFA) framework for identifying, selecting, and optimizing highly vulnerable bits in large State-Space Models (SSMs). The framework performs a two-stage pipeline:

1. **Weight Selection** — Identify candidate vulnerable weight subsets through sensitivity analysis and bit-flip perturbations.  
2. **Weight Optimization** — Reduce the subset to a minimal adversarial group while ensuring the model loss stays above a target threshold.

The framework supports several evaluation benchmarks, including ARC-Easy, HellaSwag, Lambada, MMLU, OpenBookQA, PIQA, and WinoGrande.

---

## Repository Structure

```
RAMBO/
│
├── dataset_tools/
│   ├── arceasy_tools.py
│   ├── hellaswag_tools.py
│   ├── lambada_tools.py
│   ├── mmlu_tools.py
│   ├── openbookqa_tools.py
│   ├── piqa_tools.py
│   └── winogrande_tools.py
│
├── final_weight_results/
├── plot_results/
├── sensitivity_analysis/
│
├── analysisTools.py
├── bfaTools.py
├── mmluToolSet.py
├── opt_utils.py
├── plot_utils.py
├── toolSet.py
├── utils.py
├── weight_opt.py
├── weight_sel.py
└── requirements.txt
```

---

# Installation

### 1. Create & activate a virtual environment

```bash
python3 -m venv rambo_env
source rambo_env/bin/activate        # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

RAMBO’s workflow consists of **two stages**.  
**You must run Stage 1 (weight selection) before Stage 2 (weight optimization).**

---

# Stage 1 — Weight Selection

Performs weight sensitivity scoring and selects harmful subsets of weights.

### Run:

```bash
python3 weight_sel.py     --model state-spaces/mamba-1.4b-hf     --device cuda:0     --alpha 0.25     --loss_threshold 10
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | (Required) HuggingFace model ID or local path | **None** |
| `--device` | Compute device | `"cpu"` |
| `--alpha` | Selection factor | `0.25` |
| `--loss_threshold` | Minimum loss used to filter selected weights | `10` |

Results are stored under:

```
final_weight_results/
```

---

# Stage 2 — Weight Optimization

Reduces the selected weight subset to a minimal adversarial group while keeping the loss above the threshold.

### Run:

```bash
python3 weight_opt.py     --model state-spaces/mamba-1.4b-hf     --device cuda:0     --loss_threshold 10
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | (Required) HuggingFace model ID or local path | **None** |
| `--device` | Compute device | `"cpu"` |
| `--loss_threshold` | Target loss threshold | `10` |

Optimized subsets are saved in:

```
final_weight_results/
```

---

# Datasets

Utilities for all supported benchmarks are in:

```
dataset_tools/
```

Supported datasets include:

- ARC-Easy  
- HellaSwag  
- Lambada  
- MMLU  
- OpenBookQA  
- PIQA  
- WinoGrande  

These are used internally for evaluation during weight selection and optimization.

---

# Plotting & Analysis

Use the following for generating figures:

- `plot_utils.py`
- `analysisTools.py`

Plots automatically go to:

```
plot_results/
```

---

# Model Support

Currently supports the following Mamba models:

- `state-spaces/mamba-1.4b-hf`
- `state-spaces/mamba-2.8b-hf`
- `state-spaces/mamba-370m-hf`
- `AntonV/mamba2-370m-hf`
- `AntonV/mamba2-1.3b-hf`
- `AntonV/mamba2-2.7b-hf`

---

# Reproducibility Notes

Please add you HuggingFace access token in `toolSet.py` file.