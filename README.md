# [MMM 2027] ICGF-CXRMate: Image-Conditioned Clinical Context Compression and Gated Fusion for Chest X-ray Report Generation

<!-- Paper link and final citation will be added when available. -->

## Overview

This repository provides the implementation of **ICGF-CXRMate**, a chest X-ray report generation framework that extends [CXRMate-ED](https://github.com/aehrc/cxrmate-ed) with image-conditioned clinical context compression and gated cross-attention fusion.

## Environment

```bash
conda create -n icgf python=3.10 -y
conda activate icgf
pip install -r requirement.txt
```

## Data

We follow the dataset preparation protocol of the original [CXRMate-ED](https://github.com/aehrc/cxrmate-ed), linking chest X-ray examinations from **MIMIC-CXR** with emergency-department patient records from **MIMIC-IV-ED**. Our experiments use four auxiliary sources: triage, reconciled medicines, indication, and clinical history.

## Running Experiments

First, create your local path configuration. Edit `config/paths.yaml` to set `database_dir` to the prepared dataset and `exp_dir` to the directory for checkpoints and results.

```bash
cp config/paths.example.yaml config/paths.yaml
```

### Training

**Stage 1 — Image-only training:**

```bash
dlhpcstarter -t cxrmate_ed -c config/stage_1 --train --trial 0
```

**Stage 2 — ICGF-CXRMate (image-conditioned compression and gated fusion):**

```bash
dlhpcstarter -t cxrmate_ed -c config/stage_2 --train --trial 0
```

**Stage 3 — Self-critical sequence training (SCST):**

```bash
dlhpcstarter -t cxrmate_ed -c config/stage_3 --train --trial 0
```

Stage 2 initializes from the Stage 1 checkpoint, and Stage 3 initializes from the Stage 2 checkpoint.

### Evaluation

```bash
# Stage 2 evaluation
dlhpcstarter -t cxrmate_ed -c config/stage_2 --test --trial 0

# Stage 3 evaluation
dlhpcstarter -t cxrmate_ed -c config/stage_3 --test --trial 0
```

### Baselines and Ablations

```bash
# CXRMate-ED baseline (Stage 2)
dlhpcstarter -t cxrmate_ed -c config/baselines/stage_2_cxrmate_ed --train --trial 0

# Component ablation (Stage 2)
dlhpcstarter -t cxrmate_ed -c config/ablations/stage_2_b2 --train --trial 0
```

Use the matching Stage 1 checkpoint and adjust warm-start paths when running alternative configurations.

## Inference

Generate a report for an example in the preprocessed linked test set using a compatible ICGF-CXRMate Stage 2 or Stage 3 checkpoint:

```bash
python scripts/inference.py \
  --checkpoint /path/to/icgf_checkpoint.ckpt \
  --database-dir /path/to/database/cxrmate_ed \
  --index 0 \
  --device cuda
```

## Citation
If you find this repository useful, please cite our paper:
```bibtex
@inproceedings{tran2027icgf,
  title     = {ICGF-CXRMate: Image-Conditioned Clinical Context Compression and Gated Fusion for Chest X-ray Report Generation},
  author    = {Tran, Vu Minh and Huynh, Thuc and Luu, Son T. and Bui, Doanh C.},
  booktitle = {Proceedings of the 33rd International Conference on Multimedia Modeling (MMM 2027)},
  publisher = {Springer},
  series    = {Lecture Notes in Computer Science},
  year      = {2027},
}
```
