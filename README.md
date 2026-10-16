# UPL: Uncertainty-Aware Pseudo-Labeling for Imbalanced Transductive Node Classification

[![TMLR](https://img.shields.io/badge/Published-TMLR%20(09%2F2026)-blue)](https://openreview.net/forum?id=9ovThQD3LQ)
[![OpenReview](https://img.shields.io/badge/OpenReview-9ovThQD3LQ-darkred)](https://openreview.net/forum?id=9ovThQD3LQ)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.6-3C2179)](https://pyg.org/)

Official PyTorch implementation of **“UPL: Uncertainty-aware Pseudo-labeling for Imbalanced Transductive Node Classification,”** published in *Transactions on Machine Learning Research (TMLR)* in September 2026.

**Authors:** Mohammad T. Teimuri, Zahra Dehghanian, Mahdi Mastani, Gholamali Aminian\*, and Hamid R. Rabiee\*  
**Affiliations:** Sharif University of Technology; The Alan Turing Institute  
**Paper:** [OpenReview](https://openreview.net/forum?id=9ovThQD3LQ)

---

## Overview

Graph-structured datasets often exhibit severe class imbalance, which can bias Graph Neural Networks (GNNs) toward majority classes in transductive node classification. This work derives a theoretical upper bound on population risk for imbalanced transductive node classification using transductive Rademacher complexity analysis. The analysis indicates that the bound is influenced by the number of minority-class samples and per-class node degrees.

Motivated by this result, we propose **Uncertainty-aware Pseudo-labeling (UPL)**, a method that strategically augments underrepresented classes during training.

![UPL pipeline](Input%20Graph%20v2.png)

### Key Contributions

- **Theoretical generalization bound:** Establishes an upper bound on population risk that depends on the maximum node degree within each class and the minimum node degree across the graph.
- **Topology-aware uncertainty with Selective Edge Removal (SER):** Estimates prediction stability by applying small perturbations to graph views, removing up to 100 edges per iteration, and filtering nodes with high prediction-entropy variance.
- **Band-pass pseudo-label filtering:** Uses lower and upper confidence thresholds, \(\eta_l\) and \(\eta_u\), to select intermediate-confidence nodes while reducing confirmation bias and overfitting to overly confident predictions.
- **Minority-class augmentation and balanced loss:** Restricts pseudo-labeling to nodes predicted to belong to minority classes and combines self-training with a class-weighted cross-entropy or Balanced Softmax objective.
- **Accuracy and computational efficiency:** Reports competitive balanced accuracy and macro-F1 scores on homophilic and heterophilic benchmarks. On PubMed, the paper reports 0.66 minutes for UPL versus 14.1 minutes for UNREAL across 10 runs.

## Repository Structure

```text
.
├── Main.py                 # Main entry point: graph generation, training, evaluation, and logging
├── args.py                 # Command-line arguments and hyperparameter configuration
├── utils.py                # Dataset loaders, random seeds, configuration, and helper functions
├── wandb_handler.py        # Weights & Biases tracking and local result export
├── Graph/
│   ├── graph_handler.py    # Preprocessing and imbalanced train/validation/test splits
│   └── data_config.py      # Dataset metadata and class counts
├── model/
│   ├── models.py           # GNN backbones: GCN, GAT, SAGE, GIN, and custom_GCN
│   ├── model_handler.py    # Training, SER uncertainty estimation, pseudo-label masking, and evaluation
│   ├── balanced_softmax.py # Balanced Softmax loss
│   ├── gens.py             # GraphENS auxiliary module
│   └── tam.py              # TAM auxiliary module
└── requirments.txt         # Python dependencies (filename as provided in the source text)
```

## Installation

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv upl_env
   source upl_env/bin/activate
   ```

2. Install the project dependencies. The supplied instructions specify PyTorch 2.6.0 with CUDA 12.4 and PyTorch Geometric:

   ```bash
   pip install -r requirments.txt
   ```

> **Note:** The dependency file is written as `requirments.txt` in the source material. Make sure the filename in this command matches the file in your repository.





## Experimental Results

The tables below report mean balanced accuracy (`bAcc.`) and macro-F1 scores, with standard errors over 10 repetitions and 10 pseudo-labeling iterations. Results use the **GCN** backbone with an imbalance ratio of \(\rho = 10\), except Wisconsin, which uses \(\rho = 11.63\).

### Homophilic Graph Benchmarks

| Method | Cora (bAcc.) | Cora (F1) | CiteSeer (bAcc.) | CiteSeer (F1) | PubMed (bAcc.) | PubMed (F1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 48.52 ± 1.09 | 53.58 ± 0.70 | 35.29 ± 0.34 | 22.62 ± 0.39 | 62.22 ± 0.35 | 48.99 ± 1.02 |
| Re-Weight | 65.54 ± 1.20 | 65.52 ± 0.84 | 44.52 ± 1.22 | 38.85 ± 1.62 | 70.17 ± 1.25 | 66.37 ± 1.73 |
| PC Softmax | 67.79 ± 0.92 | 67.39 ± 1.08 | 49.81 ± 1.12 | 45.55 ± 1.26 | 70.20 ± 0.60 | 68.83 ± 0.73 |
| DR-GCN | 60.17 ± 0.83 | 59.31 ± 0.97 | 42.64 ± 0.75 | 38.22 ± 1.22 | 65.51 ± 0.81 | 64.95 ± 0.63 |
| GraphSMOTE | 62.66 ± 0.83 | 61.76 ± 0.96 | 34.26 ± 0.89 | 28.31 ± 1.48 | 68.94 ± 0.89 | 64.17 ± 1.43 |
| Balanced Softmax | 67.75 ± 1.09 | 66.47 ± 1.27 | 52.09 ± 1.61 | 48.55 ± 2.01 | 71.78 ± 0.82 | 71.18 ± 0.94 |
| GraphENS + TAM | 69.42 ± 0.90 | 68.70 ± 1.08 | 53.10 ± 1.82 | 50.67 ± 2.28 | 71.70 ± 0.95 | 69.01 ± 1.72 |
| BAT1 | 72.32 ± 0.52 | 71.10 ± 0.55 | 59.23 ± 0.90 | 58.48 ± 0.87 | 73.37 ± 0.45 | 72.60 ± 0.48 |
| UNREAL | **78.33 ± 1.04** | **76.44 ± 1.06** | 65.63 ± 1.38 | 64.94 ± 1.38 | 75.35 ± 1.41 | 73.65 ± 1.43 |
| **UPL (Ours)** | 76.16 ± 0.76 | 74.53 ± 0.85 | **68.18 ± 0.27** | **67.78 ± 0.25** | **77.50 ± 0.43** | **77.19 ± 0.44** |

### Heterophilic Graph Benchmarks

| Method | Chameleon (bAcc.) | Chameleon (F1) | Squirrel (bAcc.) | Squirrel (F1) | Wisconsin (bAcc.) | Wisconsin (F1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 57.01 ± 0.53 | 55.61 ± 0.67 | 38.07 ± 0.57 | 34.81 ± 0.69 | 34.51 ± 2.31 | 31.45 ± 2.66 |
| Balanced Softmax | 58.60 ± 1.06 | 58.20 ± 1.00 | 41.81 ± 0.64 | 41.27 ± 0.71 | 44.44 ± 3.46 | 41.29 ± 2.88 |
| GraphENS + TAM | 40.98 ± 0.85 | 38.93 ± 0.84 | 27.98 ± 0.90 | 24.04 ± 0.70 | 41.38 ± 2.78 | 34.56 ± 2.22 |
| UNREAL | 58.18 ± 0.94 | 56.98 ± 1.03 | **45.65 ± 1.08** | **44.78 ± 1.07** | 41.26 ± 5.19 | 32.14 ± 2.15 |
| **UPL (Ours)** | **61.19 ± 0.66** | **60.71 ± 0.71** | 43.37 ± 0.25 | 42.82 ± 0.07 | **48.70 ± 2.88** | **42.72 ± 2.38** |

Additional experiments using `GAT`, `GraphSAGE`, and `GIN` backbones; the `Amazon Computers`, `Amazon Photo`, and `Flickr` datasets; and imbalance ratios \(\rho = 2, 5, 20\) are reported in Appendix H of the paper.

## Citation

If you use this repository or paper in your research, please cite the work:

```bibtex
@article{teimuri2026upl,
  title   = {UPL: Uncertainty-aware Pseudo-labeling for Imbalanced Transductive Node Classification},
  author  = {Teimuri, Mohammad T. and Dehghanian, Zahra and Mastani, Mahdi and Aminian, Gholamali and Rabiee, Hamid R.},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  month   = {09},
  url     = {https://openreview.net/forum?id=9ovThQD3LQ}
}
```
