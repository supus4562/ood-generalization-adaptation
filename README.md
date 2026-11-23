# Out-of-Distribution Generalization, Domain Adaptation & Open-Set Recognition Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-0.19%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive empirical research benchmark evaluating representation robustness when standard IID (independent and identically distributed) assumptions break down. Covers inductive bias probing under style transfer, Unsupervised Domain Adaptation (UDA), Sharpness-Aware Domain Generalization (DG), and state-of-the-art Open-Set Recognition (OSR).

---

## 📌 Benchmark Suites & Methodology

| Suite | Problem Formulation | Core Methods & Evaluated Architectures |
| :--- | :--- | :--- |
| **1. Inductive Biases & Representations** | Shape vs. texture bias under AdaIN style transfer cue-conflicts | ResNet-50, ViT-B/16, and CLIP (Zero-shot visual representations) |
| **2. Unsupervised Domain Adaptation (UDA)** | Covariate shift alignment between Photo $\to$ Sketch on PACS | Maximum Mean Discrepancy (DAN), Gradient Reversal (DANN), Conditional Adversarial (CDAN) |
| **3. Domain Generalization (DG)** | Zero-shot generalization to unseen target domains | Cross-source MMD (DAN-DG), Sharpness-Aware Minimization (SAM, $\rho \in \{0.01, 0.05, 0.1\}$) |
| **4. Open-Set Recognition (OSR)** | Rejecting out-of-distribution unknown classes (CIFAR-10 vs CIFAR-100) | PROSER (Placeholder class modeling), GCSC, and post-hoc detectors (Energy, Mahalanobis, MSP, MLS) |

---

## 📁 Repository Structure

```text
├── common/                    # Shared dataset loaders, transforms, and logging utilities
├── task1/                     # Inductive Biases & Feature Representations
│   ├── scripts/run_task1.py   # AdaIN cue-conflict style transfer and evaluation pipeline
│   └── results/figures/       # Shape vs texture bias trade-off plots
├── task2/                     # Unsupervised Domain Adaptation (UDA)
│   ├── configs/               # source_only.yaml, dan.yaml, dann.yaml, cdan.yaml
│   ├── models/                # Backbones, gradient reversal layers, domain discriminators
│   ├── train.py               # UDA alignment trainer
│   └── evaluate_final.py      # Cross-domain target accuracy evaluator
├── task3/                     # Domain Generalization (DG)
│   ├── configs/               # dan_dg.yaml, sam.yaml (SAM sharpness radii)
│   ├── sam.py                 # Sharpness-Aware Minimization optimizer implementation
│   ├── train.py               # Multi-source domain trainer
│   └── evaluate_sketch.py     # Unseen Sketch domain generalization evaluator
├── task4/                     # Open-Set Recognition (OSR)
│   ├── configs/               # vanilla.yaml, gcsc.yaml, proser.yaml
│   ├── models/                # PROSER placeholder head and classifier architectures
│   ├── train.py               # Known-class training with regularized placeholder objectives
│   ├── extract_outputs.py     # Penultimate feature and logit extraction (TF32 accelerated)
│   ├── evaluate_osr.py        # Post-hoc score evaluation (AUROC, FPR95, OSCR)
│   └── run_task4.sh           # End-to-end OSR execution pipeline
├── utilities/                 # Automated high-speed dataset download mirrors
└── requirements.txt           # Environment specifications
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/supus4562/ood-generalization-adaptation.git
cd ood-generalization-adaptation
pip install -r requirements.txt
```

---

## 🔬 Benchmark Execution

### 1. Inductive Biases and Feature Representations (AdaIN)
```bash
python task1/scripts/run_task1.py   --data_root ./data/oxford-iiit-pet   --output_dir task1/results
```

### 2. Unsupervised Domain Adaptation (DAN, DANN, CDAN)
```bash
# Train Source-Only Baseline, DAN, DANN, and CDAN on PACS
python task2/train.py --config task2/configs/source_only.yaml --pacs_root <pacs_path>
python task2/train.py --config task2/configs/dan.yaml        --pacs_root <pacs_path>
python task2/train.py --config task2/configs/dann.yaml       --pacs_root <pacs_path>
python task2/train.py --config task2/configs/cdan.yaml       --pacs_root <pacs_path>

# Evaluate target domain transfer accuracy
python task2/evaluate_final.py --pacs_root <pacs_path> --output_dir task2/results
```

### 3. Domain Generalization with SAM
```bash
# Multi-source training with Sharpness-Aware Minimization
python task3/train.py --config task3/configs/dan_dg.yaml --pacs_root <pacs_path>
python task3/train.py --config task3/configs/sam.yaml    --pacs_root <pacs_path>

# Generalization evaluation on unseen Sketch domain
python task3/evaluate_sketch.py --pacs_root <pacs_path> --output_dir task3/results
```

### 4. Open-Set Recognition (PROSER & Post-Hoc Detectors)
```bash
# Run full pipeline (Training, Feature Extraction, and AUROC/FPR95 Evaluation)
bash task4/run_task4.sh ./data/cifar task4/results

# Or execute step-by-step:
python task4/train.py --config task4/configs/proser.yaml  --data_root ./data/cifar
python task4/extract_outputs.py --data_root ./data/cifar
python task4/evaluate_osr.py    --data_root ./data/cifar --output_dir task4/results
```

---

## 🔬 Key Empirical Insights

1. **Vision Transformers vs. CNNs on Cue Conflicts**:
   Vision Transformers exhibit significantly higher shape bias and robustness to high-frequency texture corruption compared to ResNet architectures, with CLIP visual representations demonstrating the strongest semantic invariance under extreme AdaIN style shifts.

2. **Domain Invariance vs Discriminability (DANN vs CDAN)**:
   While standard DANN (Ganin et al.) aligns global marginal distributions, Conditional Domain Adversarial Networks (CDAN) condition domain discrimination on classifier predictions, preventing mode collapse across semantically complex multi-class boundaries.

3. **Placeholder Modeling (PROSER) for Unknown Rejection**:
   By learning dedicated placeholder representations to bound the known class manifolds, PROSER substantially suppresses overconfident activations on out-of-distribution unknown samples compared to maximum softmax probability (MSP) baselines.

---

## 📜 Attributions & References

- **AdaIN Style Transfer**: Huang & Belongie, *"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"*, ICCV 2017.
- **DANN**: Ganin et al., *"Domain-Adversarial Training of Neural Networks"*, JMLR 2016.
- **CDAN**: Long et al., *"Conditional Adversarial Domain Adaptation"*, NeurIPS 2018.
- **SAM**: Foret et al., *"Sharpness-Aware Minimization for Efficiently Improving Generalization"*, ICLR 2021.
- **PROSER**: Zhou et al., *"Learning Placeholders for Open-Set Recognition"*, CVPR 2021.
- **CLIP**: Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"*, ICML 2021.

---

## 📄 License
This project is licensed under the MIT License.
