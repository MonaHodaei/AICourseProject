# RandLA-Net: Semantic Segmentation of Bridge Components and Highway Infrastructure from Mobile LiDAR Data

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![CUDA 12](https://img.shields.io/badge/CUDA-12-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Downloads — Data & Checkpoints](#downloads--data--checkpoints)
3. [Environment Setup](#environment-setup)
4. [Installation Guide](#installation-guide)
5. [Project Structure](#project-structure)
6. [Workflow Explanation](#workflow-explanation)
7. [Code Breakdown](#code-breakdown)
8. [Usage Instructions](#usage-instructions)
9. [Results](#results)
10. [Data Availability](#data-availability)
11. [Code Attribution](#code-attribution)

---

## Project Overview

This project implements a semantic segmentation framework for classifying **bridge components and highway infrastructure** from large-scale mobile LiDAR point clouds. Data is acquired using the **Purdue Wheel-Based Mobile Mapping System (MMS)** along Indiana interstate highways.

### Key Contributions

**Deep Learning Backbone — RandLA-Net:** RandLA-Net is adopted as the core architecture for efficient processing of large-scale, unstructured point cloud data. It uses an encoder-decoder structure with Local Feature Aggregation (LFA) modules combining random sampling, local spatial encoding, and attentive pooling.

**Two-Stage Transfer Learning:** The model is first pretrained on the large-scale [Semantic3D](https://semantic3d.ethz.ch/) benchmark, then fine-tuned on a domain-specific Purdue bridge and highway dataset. This substantially reduces the need for extensive manual annotation.

**Structured Annotation Workflow:** A consistent labeling pipeline using [CloudCompare](https://www.cloudcompare.org/) ensures dataset reliability across nine semantic classes covering both bridge components and the highway infrastructure.

### Semantic Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Unlabeled | Unannotated points |
| 1 | Bridge — Deck, Beam & Girder | Horizontal bridge components |
| 2 | Bridge — Abutment & Wing Wall | Vertical end supports |
| 3 | Bridge — Pier | Vertical mid-span supports |
| 4 | Man-made Terrain | Paved road surfaces |
| 5 | Natural Terrain | Grass and drainage ditches |
| 6 | Vegetation | Trees and shrubs |
| 7 | Buildings | Retaining walls, noise barriers |
| 8 | Remaining Hardscape | Guardrails, signs, utility poles |
| 9 | Scanning Artifacts | Spurious points from moving objects |

### Performance Summary

The framework was evaluated on four test tiles (~120 m each) from the I-465 highway near Indianapolis, Indiana (April 2023 dataset):

| Tile | Overall Accuracy | Bridge Deck F1 |
|------|-----------------|----------------|
| Tile 1 | 91% | 0.98 |
| Tile 2 | 91% | — (no bridge) |
| Tile 3 | 83% | 0.85 |
| Tile 4 | 95% | — (no bridge) |

---

## Downloads — Data & Checkpoints

Due to data sharing restrictions (INDOT–Purdue University research agreement) and file size, the dataset and model checkpoints are **not included in this repository**. They are hosted on Google Drive and must be downloaded manually before running the pipeline.

| Item | Description | Link |
|------|-------------|------|
| `randlanet_semantic3d.pth` | Stage 1 pretrained checkpoint (Semantic3D benchmark) | [GOOGLE DRIVE LINK] |
| `randlanet_Purdue.pth` | Stage 2 final checkpoint (fine-tuned on Purdue bridge components & highway infrusturcture dataset) | [GOOGLE DRIVE LINK] |
| Trial test tile (`Test.txt`) | One preprocessed, labeled test tile for demo purposes | [GOOGLE DRIVE LINK] |

### Setup after downloading

1. Place the checkpoint files into the `checkpoints/` folder:
   ```
   checkpoints/
   ├── randlanet_semantic3d.pth
   └── randlanet_Purdue.pth
   ```

2. Place the trial test tile into the `Dataset/Test/` folder:
   ```
   Dataset/Test/
   └── Test.txt
   ```

The repo is then ready to run from **Step 3 (inference)** onwards using the trial tile and the provided Purdue checkpoint. To run the full pipeline from scratch including training, you will need to supply your own labeled dataset.

---

## Environment Setup

The environment is managed with **Conda** using the provided `environment.yml` file. A CUDA-capable GPU is required for training and inference.

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA 12-compatible GPU (recommended: NVIDIA with ≥8 GB VRAM)
- Linux OS (tested on Ubuntu)

### Create and Activate the Environment

```bash
# Create the conda environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate randlanet
```

> **Note:** The `prefix` field in `environment.yml` is set to the original author's path. You can safely ignore any prefix-related warnings — the environment will install correctly regardless.

### Verify the Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## Installation Guide

After activating the conda environment, install any remaining dependencies and build the C++ wrappers used by RandLA-Net's KNN operations.

### Step 1 —  Activate Environment

```bash
conda activate randlanet
```

---

### Step 2 — Build C++ Wrappers

Run this **before** installing `torch-points-kernels`, while `numpy=1.23.5` is still intact:

```bash
cd utils/cpp_wrappers
sh compile_wrappers.sh
cd ../..
```

---

### Step 3 — Install torch-points-kernels

```bash
pip install torch-points-kernels==0.7.0
```

> **Note:** This will downgrade numpy to `1.19.5`. That is expected — fix it in the next step.

---

### Step 4 — Restore numpy

```bash
pip install "numpy==1.23.5" --force-reinstall
```

This restores numpy compatibility with pandas, which is required for the preprocessing notebook.

---
### Step 5 — Install Kernel

```bash
pip install notebook
```

---
### Step 6 — Verify Installation

```bash
python -c "
import numpy as np
import pandas as pd
import torch
from torch_points_kernels import knn
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('torch_points_kernels: OK')
"
```


---

## Project Structure

```
RandLA-Net-pytorch/
├── environment.yml                        # Conda environment specification
├── model.py                               # RandLA-Net architecture definition
│
├── point_cloud_processing.ipynb           # Step 1: Preprocessing pipeline
├── train.ipynb                            # Step 2a: Train from scratch on Purdue data
├── transferlearning.ipynb                 # Step 2b: Fine-tune from Semantic3D weights
├── test.ipynb                             # Step 3: Run inference on test tiles
├── quantitative_Evaluation.ipynb          # Step 4: Compute evaluation metrics
│
├── Dataset/
│   ├── Raw/                               # Raw MMS point cloud input (.txt)
│   ├── Preparation/
│   │   ├── 0.Downsampled/                 # After 5 cm distance-based downsampling
│   │   ├── 1.Tiled/                       # After tiling into 120 m segments
│   │   ├── 2.Localized/                   # After zero-centering (local coordinates)
│   │   └── 3.Ordered/                     # After column reordering for model input
│   ├── Train/                             # Labeled training tiles — not publicly available (INDOT–Purdue agreement); see Downloads for trial sample
│   └── Test/                              # Labeled test tiles — trial sample available on Google Drive (see Downloads)
│
├── checkpoints/
│   ├── randlanet_semantic3d.pth            # Stage 1: weights pretrained on the Semantic3D benchmark — download from Google Drive (see Downloads)
│   └── randlanet_Purdue.pth               # Stage 2: final checkpoint after transfer learning on Purdue dataset — download from Google Drive (see Downloads)
│
├── results/
│   ├── accuracy_report.txt                # Per-class metrics summary
│   ├── confusion_matrix.txt               # Confusion matrix
│   ├── Purdue/                            # Predictions using Purdue fine-tuned model
│   └── Semantic3D/                        # Predictions using Semantic3D-only model
│
├── runs/
│   ├── training/                          # TensorBoard logs for from-scratch training
│   └── transfer_learning/                 # TensorBoard logs for transfer learning
│
└── utils/
    ├── metrics.py                         # Accuracy and IoU computation utilities
    ├── ply.py                             # PLY file I/O utilities
    └── cpp_wrappers/                      # Optional C++ KNN acceleration
```

---

## Workflow Explanation

The project follows a four-stage pipeline. Run each step in sequence:

```
[Raw LiDAR .txt]
       │
       ▼
 ┌─────────────────────────────────┐
 │  STEP 1: point_cloud_processing │  Downsample → Tile → Localize → Reorder
 └─────────────────────────────────┘
       │
       ▼ (labeled tiles in Dataset/Train/)
 ┌─────────────────────────────────┐
 │  STEP 2a: train.ipynb           │  Train RandLA-Net from scratch (Purdue data)
 │        ── OR ──                 │
 │  STEP 2b: transferlearning.ipynb│  Fine-tune from Semantic3D pretrained weights
 └─────────────────────────────────┘
       │
       ▼ (checkpoint: checkpoints/randlanet_Purdue.pth)
 ┌─────────────────────────────────┐
 │  STEP 3: test.ipynb             │  Run inference → per-point class predictions
 └─────────────────────────────────┘
       │
       ▼ (prediction .txt + .ply files in results/)
 ┌─────────────────────────────────┐
 │  STEP 4: quantitative_Evaluation│  Compare predictions vs. ground truth
 └─────────────────────────────────┘
       │
       ▼ (accuracy_report.txt, confusion_matrix.txt)
```

---

## Code Breakdown

### `model.py` — RandLA-Net Architecture

**Purpose:** Defines the complete RandLA-Net model for large-scale point cloud semantic segmentation.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `SharedMLP` | Shared 2D convolution block with optional batch normalization and activation |
| `LocalSpatialEncoding` | Encodes relative 3D position between a point and its K nearest neighbors into a 10-dim feature vector |
| `AttentivePooling` | Aggregates neighbor features using learned attention scores |
| `LocalFeatureAggregation (LFA)` | Full LFA module: two rounds of LSE + AttentivePooling in a dilated residual block |
| `RandLANet` | Full encoder-decoder with 5 encoding layers (progressive downsampling by factor 4) and 5 decoding layers with skip connections; outputs per-point class logits |

**Inputs:** Point cloud tensor `(B, N, D_IN)` — batch of N points with D_IN features (X, Y, Z, Intensity, PointSourceID).

**Output:** Per-point class logit tensor `(B, N, NUM_CLASSES)`.

**Key Parameters:**

```python
D_IN          = 5      # Input feature dimensions (X, Y, Z, Intensity, PointSourceID)
NUM_CLASSES   = 10     # Number of semantic classes (including unlabeled)
NUM_NEIGHBORS = 16     # K nearest neighbors per point
DECIMATION    = 4      # Downsampling ratio per encoder layer
NUM_LAYERS    = 5      # Number of encoder/decoder stages
NUM_POINTS    = 20480  # Points sampled per training tile (must be divisible by 4^5)
```

---

### `point_cloud_processing.ipynb` — Preprocessing Pipeline

**Purpose:** Transforms raw MMS LiDAR data into model-ready tiles through four sequential steps.

| Cell | Step | Description |
|------|------|-------------|
| Cell 1 | Downsampling | Memory-efficient distance-based downsampling with 5 cm minimum point spacing. Reads `Dataset/Raw/PC.txt`, outputs `Dataset/Preparation/0.Downsampled/PC_DS5cm.txt` |
| Cell 2 | Tiling | Splits the downsampled cloud into 120 m rectangular segments along the GPS-time direction. Outputs individual tile files to `Dataset/Preparation/1.Tiled/` |
| Cell 3 | Localization | Zero-centers each tile (subtracts tile origin) to improve numerical stability. Saves tile origins to `all_tiles_origins.csv`. Outputs to `Dataset/Preparation/2.Localized/` |
| Cell 4 | Column Reordering | Selects and reorders columns to the format expected by the model: `[X, Y, Z, Intensity, PointSourceID, Label]`. Outputs to `Dataset/Preparation/3.Ordered/` |

**Required Input:** `Dataset/Raw/PC.txt` — whitespace-delimited point cloud file with columns `[X, Y, Z, Intensity, ..., PointSourceID, GPSTime]`.

**Expected Output:** Processed tile files in `Dataset/Preparation/3.Ordered/`, ready to be placed into `Dataset/Train/` or `Dataset/Test/` after manual annotation in CloudCompare.

---

### `train.ipynb` — Training from Scratch

**Purpose:** Trains a RandLA-Net model from random initialization on the labeled Purdue bridge components and highway infrastructure dataset.

**Required Inputs:**
- Labeled tile `.txt` files in `Dataset/Train/` — format: `X Y Z Intensity PointSourceID Label`
- No pretrained checkpoint required (starts fresh or resumes from `CHECKPOINT_PATH`)

**Key Configuration (edit at top of notebook):**

```python
DATA_DIR        = Path("Dataset/Train")
LOGS_DIR        = Path("runs/training")
CHECKPOINT_PATH = None        # Path to .pth to resume, or None to start fresh

D_IN            = 5           # Feature dimensions
NUM_CLASSES     = 10
NUM_NEIGHBORS   = 16
DECIMATION      = 4
NUM_LAYERS      = 5
NUM_POINTS      = 20480
EPOCHS          = 50
ADAM_LR         = 1e-2
SCHEDULER_GAMMA = 0.95
VAL_SPLIT       = 0.2
BATCH_SIZE      = 1
```

**Processing:** At each epoch, the DataLoader randomly samples 20,480 points from each tile, computes KNN graphs, and passes batches through RandLA-Net. Cross-entropy loss is minimized with Adam + exponential LR decay. Training and validation metrics (loss, accuracy, mIoU) are logged to TensorBoard.

**Expected Output:**
- Saved model checkpoint `.pth` files every `SAVE_FREQ` epochs
- TensorBoard logs in `runs/training/`

**Monitor Training:**
```bash
tensorboard --logdir runs/training
```

---

### `transferlearning.ipynb` — Transfer Learning (Recommended)

**Purpose:** Fine-tunes a RandLA-Net model pretrained on Semantic3D onto the Purdue bridge components and highway infrastructure dataset using a two-stage transfer learning strategy.

**Required Inputs:**
- `checkpoints/randlanet_semantic3d.pth` — pretrained Semantic3D weights
- Labeled tile `.txt` files in `Dataset/Train/`

**Two-Stage Strategy:**

Stage 1 — Weight loading: Encoder and decoder weights are loaded from the Semantic3D checkpoint. The input projection layer (`fc0`) and output classification head (`fc1`) are re-initialized to match the new feature dimensions (`D_IN=5`) and class count (`NUM_CLASSES=10`).

Stage 2 — Fine-tuning: All layers are trained jointly on the Purdue dataset, allowing the model to adapt its general geometric feature representations to the domain-specific bridge infrastructure classes.

**Key Difference from `train.ipynb`:** The transfer learning notebook initializes from Semantic3D weights rather than random initialization, leading to faster convergence and better performance on underrepresented classes with limited training data.

**Expected Output:**
- Fine-tuned checkpoint saved as `checkpoints/randlanet_Purdue.pth`
- TensorBoard logs in `runs/transfer_learning/`

**Monitor Training:**
```bash
tensorboard --logdir runs/transfer_learning
```

---

### `test.ipynb` — Inference / Prediction

**Purpose:** Applies the trained RandLA-Net model to unlabeled or labeled test tiles and saves per-point predictions.

**Required Inputs:**
- Test tile `.txt` files in `Dataset/Test/` — format: `X Y Z Intensity PointSourceID [Label]`
- A trained model checkpoint (e.g., `checkpoints/randlanet_Purdue.pth`)

**Key Configuration:**

```python
RANDLA_ROOT     = Path(".")
TEST_DIR        = Path("Dataset/Test")
RESULTS_DIR     = Path("results/Purdue")
CHECKPOINT_PATH = Path("checkpoints/randlanet_Purdue.pth")

NUM_CLASSES     = 10
NUM_NEIGHBORS   = 16
DECIMATION      = 4
NUM_LAYERS      = 5
D_IN            = 5
```

**Processing:** The model runs inference tile-by-tile. Since test tiles can contain millions of points (2.5–3.9 million per tile in this study), predictions are accumulated in batches of 20,480 points. Final labels are assigned by majority vote or direct argmax over the logits.

**Expected Output (per tile):**
- `results/Purdue/<tile_name>_pred.txt` — point cloud with predicted class labels
- `results/Purdue/<tile_name>_pred.ply` — PLY format for visualization in CloudCompare or MeshLab

---

### `quantitative_Evaluation.ipynb` — Evaluation Metrics

**Purpose:** Computes quantitative performance metrics by comparing model predictions against ground truth labels.

**Required Inputs:**
- Prediction `.txt` file (output of `test.ipynb`)
- Ground truth `.txt` file (labeled tile from `Dataset/Test/`)

**Processing:**
- Aligns prediction and ground truth point sets (uses KDTree matching if point counts differ)
- Computes the confusion matrix
- Derives per-class metrics: Precision, Recall, F1-score, IoU (TPrate)
- Computes Overall Accuracy and mean IoU across non-empty classes

**Expected Output:**
- Printed per-class metrics table in the notebook
- `results/accuracy_report.txt` — saved performance summary
- `results/confusion_matrix.txt` — saved confusion matrix

---

### `utils/metrics.py` — Evaluation Utilities

**Purpose:** Provides helper functions for computing accuracy and IoU during training.

**Functions:**
- `accuracy(predictions, labels)` — computes overall point-wise classification accuracy
- `intersection_over_union(predictions, labels, num_classes)` — computes per-class IoU

> **Source:** `metrics.py` is borrowed directly from [aRI0U/RandLA-Net-pytorch](https://github.com/aRI0U/RandLA-Net-pytorch/tree/master/utils).

---

### `utils/ply.py` — PLY File I/O

**Purpose:** Reads and writes PLY format point cloud files for visualization in tools like CloudCompare, MeshLab, or Open3D.

> **Source:** `ply.py` is borrowed directly from [aRI0U/RandLA-Net-pytorch](https://github.com/aRI0U/RandLA-Net-pytorch/tree/master/utils).

---

## Usage Instructions

### 1. Preprocess Raw LiDAR Data

Place your raw MMS point cloud (whitespace-delimited `.txt` with columns `X Y Z Intensity ... PointSourceID GPSTime`) at:

```
Dataset/Raw/PC.txt
```

Open and run all cells in `point_cloud_processing.ipynb`:

```bash
jupyter notebook point_cloud_processing.ipynb
```

After processing, manually annotate the tiles in CloudCompare and place labeled files in `Dataset/Train/` and `Dataset/Test/`.

### 2a. Train from Scratch

```bash
jupyter notebook train.ipynb
```

Edit the `DATA_DIR` and `LOGS_DIR` paths at the top of the notebook to match your system, then run all cells.

### 2b. Fine-Tune from Semantic3D Weights (Recommended)

```bash
jupyter notebook transferlearning.ipynb
```

Ensure `checkpoints/randlanet_semantic3d.pth` is present (download from Google Drive — see [Downloads](#downloads--data--checkpoints)), edit paths as needed, then run all cells. The fine-tuned model will be saved to `checkpoints/randlanet_Purdue.pth`.

### 3. Run Inference

```bash
jupyter notebook test.ipynb
```

Set `CHECKPOINT_PATH` to your trained model (e.g., `checkpoints/randlanet_Purdue.pth`) and `TEST_DIR` to the folder containing your test tiles. Run all cells to generate prediction files in `results/Purdue/`.

### 4. Evaluate Results

```bash
jupyter notebook quantitative_Evaluation.ipynb
```

Set the paths to the prediction and ground truth files, then run all cells. Metrics will be printed inline and saved to `results/accuracy_report.txt`.

### 5. Visualize Predictions

Open any `_pred.ply` file from `results/Purdue/` in [CloudCompare](https://www.cloudcompare.org/) or [MeshLab](https://www.meshlab.net/). Color-code by the scalar field containing the predicted class label to inspect segmentation quality.

---

## Results

Training and validation curves can be visualized with TensorBoard:

```bash
# For from-scratch training
tensorboard --logdir runs/training

# For transfer learning
tensorboard --logdir runs/transfer_learning
```

Quantitative test results for the four I-465 highway tiles:

| Tile | Points | Overall Accuracy |
|------|--------|-----------------|
| Tile 1 | 2,915,047 | 91% |
| Tile 2 | 2,504,217 | 91% |
| Tile 3 | 3,936,656 | 83% |
| Tile 4 | 2,694,649 | 95% |

Training convergence (700 epochs, transfer learning):

| Metric | Training | Validation |
|--------|----------|------------|
| IoU | 0.83 | 0.75 |
| Accuracy | 0.91 | 0.83 |
| Loss | 0.12 | 0.17 |

---



## Data Availability

The LiDAR dataset used in this project was collected as part of a collaborative research project between the **Indiana Department of Transportation (INDOT)** and **Purdue University**. Due to the terms of this partnership, the full dataset cannot be publicly shared.

A small trial sample — one preprocessed, labeled test tile — is made available via Google Drive for demonstration and reproducibility purposes (see [Downloads](#downloads--data--checkpoints)). This tile is sufficient to run Steps 3 and 4 of the pipeline (inference and evaluation) using the provided Purdue checkpoint.

If you require access to additional data for research purposes, please contact my email.

---

## Code Attribution

This codebase was developed as part of AI course project by Mona Hodaei. The following attribution applies to individual components:

| File | Origin |
|------|--------|
| `utils/metrics.py` | Borrowed from [aRI0U/RandLA-Net-pytorch](https://github.com/aRI0U/RandLA-Net-pytorch/tree/master/utils) |
| `utils/ply.py` | Borrowed from [aRI0U/RandLA-Net-pytorch](https://github.com/aRI0U/RandLA-Net-pytorch/tree/master/utils) |
| `model.py` | Originally written by the author, inspired by [aRI0U/RandLA-Net-pytorch](https://github.com/aRI0U/RandLA-Net-pytorch) and the [Open3D-ML](https://github.com/isl-org/Open3D-ML) implementation of RandLA-Net; edited with assistance from LLM tools and online references |
| All notebooks (`point_cloud_processing.ipynb`, `train.ipynb`, `transferlearning.ipynb`, `test.ipynb`, `quantitative_Evaluation.ipynb`) | Written by the author; edited and refined with assistance from online tools and references |

---

## Acknowledgments

Data was collected using the Purdue Wheel-Based Mobile Mapping System along Indiana interstate highways, as part of a project with the Indiana Department of Transportation (INDOT). The RandLA-Net architecture is based on [Hu et al., CVPR 2020](https://arxiv.org/abs/1911.11236). Pretraining was performed on the [Semantic3D](https://semantic3d.ethz.ch/). Utility functions (`metrics.py`, `ply.py`) and structural inspiration for `model.py` are credited to [aRI0U/RandLA-Net-pytorch](https://github.com/aRI0U/RandLA-Net-pytorch/tree/master/utils) and [Open3D-ML](https://github.com/isl-org/Open3D-ML).
