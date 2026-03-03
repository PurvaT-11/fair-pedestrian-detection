# Fair Pedestrian Detection with YOLOv8

A fairness-aware object detection system addressing height-based bias in autonomous vehicle pedestrian detection. Built using the BDD100K dataset and Ultralytics YOLOv8, this project implements novel auditing techniques and bias mitigation strategies to ensure equitable performance across pedestrians of different heights.

## Table of Contents

- [Motivation](#motivation)
- [Key Features](#key-features)
- [Technical Approach](#technical-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)

## Motivation

Recent research has revealed significant disparities in pedestrian detection accuracy based on height. Studies show a 23% higher false-negative rate for pedestrians under 140 cm (such as children) compared to taller individuals. This disparity poses serious real-world safety concerns, particularly in autonomous vehicle systems where failure to detect shorter pedestrians can lead to catastrophic consequences.

The problem stems from several factors:
- **Dataset imbalance**: Training datasets contain disproportionately fewer examples of shorter pedestrians
- **Scale bias**: Detection models inherently perform better on larger objects due to more available features
- **Distance confounding**: Shorter pedestrians are often confused with distant adults, leading to inconsistent detection behavior

This project addresses these challenges through systematic auditing and targeted mitigation strategies.

## Key Features

- **Height-stratified performance auditing** with automated bias detection
- **Instance-level height weighting** for fairness-aware training
- **Comprehensive fairness metrics** including Height Detection Rate (HDR) and confidence score analysis
- **End-to-end pipeline** from data conversion to model deployment
- **Reproducible experiments** with detailed logging and result tracking

## Technical Approach

### 1. Bias Auditing Framework

The project implements a comprehensive auditing system to quantify height-based bias in pedestrian detection models.

#### Detection Export and Enrichment

The auditing process begins with systematic detection on a sampled validation set:

```python
# scripts/export_detections.py
- Samples 500 images from BDD100K validation set
- Runs inference with YOLOv8
- Exports per-instance detection metadata (bounding box dimensions, confidence scores)
```

Each detection is enriched with computed height metrics:

```python
# scripts/enrich_predictions.py
- Calculates bounding box height (ymax - ymin)
- Assigns height group labels (short: <100px, tall: >=100px)
- Generates enriched dataset for analysis
```

#### Height-Stratified Analysis

The audit script performs group-wise statistical analysis:

```python
# scripts/audit_height_bias.py
- Computes mean confidence scores per height group
- Calculates detection counts and rates
- Identifies performance disparities
```

**Novel Aspect**: Unlike traditional fairness audits that focus on protected attributes (race, gender), this approach introduces height as a continuous bias dimension in object detection. The pixel-based height threshold (100px) is derived empirically from the distribution of pedestrian sizes in driving scenarios, corresponding approximately to the 140cm physical height identified in safety research.

### 2. Height-Weighted Training

The primary bias mitigation technique is instance-level sample weighting during training.

#### Training Data Reweighting

The system creates a height-weighted training configuration:

```yaml
# dataset/bdd100k/bdd100k_height_weighted.yaml
train: height_weighted_train.txt  # Modified training manifest with instance weights
```

The weighting strategy works as follows:

1. **Height Extraction**: Parse ground truth bounding boxes to compute pedestrian heights
2. **Weight Assignment**: Apply inverse frequency weighting to balance height distribution
   - Shorter pedestrians (h < threshold) receive higher weights
   - Weights are normalized to prevent destabilizing training
3. **Instance Sampling**: Modified data loader samples instances proportional to assigned weights

**Novel Aspect**: While class-level reweighting is common in classification tasks, this project applies fine-grained instance-level reweighting based on a continuous physical attribute (height) within a single class (pedestrian). This is particularly challenging because:
- It requires preserving spatial detection accuracy while adjusting for fairness
- It operates within the constraints of the YOLO detection framework
- It balances fairness objectives with overall detection performance

#### Training Configuration

```python
# scripts/train.py
model.train(
    data="dataset/bdd100k/bdd100k_height_weighted.yaml",  # Height-weighted data
    epochs=1,
    imgsz=320,
    batch=8,
    classes=[0]  # Pedestrian class only
)
```

The training process is optimized for:
- **Computational efficiency**: Reduced image size (320px) and batch size (8) enable rapid iteration
- **Class focus**: Single-class training eliminates confounding factors from multi-object detection
- **Reproducibility**: Fixed random seeds and deterministic operations ensure consistent results

### 3. Fairness-Aware Evaluation

The project implements custom metrics to quantify fairness alongside traditional detection performance.

#### Traditional Detection Metrics

- **mAP (mean Average Precision)**: Overall detection quality across confidence thresholds
- **Precision/Recall**: Standard object detection performance measures
- **Box Loss**: Bounding box regression accuracy (IoU-based)
- **Classification Loss**: Confidence score calibration

#### Fairness-Specific Metrics

**Height Detection Rate (HDR)**:
```
HDR_group = (Detected pedestrians in group) / (Total pedestrians in group)
```

**Height Detection Ratio**:
```
HDR_ratio = HDR_short / HDR_tall
```
Ideal value is 1.0, indicating equitable detection across height groups.

**Confidence Score Disparity**:
```
Confidence_gap = mean(confidence_tall) - mean(confidence_short)
```
Measures systematic differences in model certainty across groups.

**Novel Aspect**: The Height Detection Ratio provides a single interpretable metric for fairness evaluation, analogous to demographic parity in classification but adapted for object detection. The use of confidence score disparity as a secondary metric captures subtle biases that may not manifest in detection rates alone (e.g., a model may detect both groups equally but be less confident about shorter pedestrians, indicating learned uncertainty).

### 4. Dataset Processing

#### BDD100K to YOLO Conversion

```python
# scripts/convert_bdd100k_to_yolo.py
- Parses BDD100K JSON annotations
- Converts polygon/box annotations to YOLO format (normalized coordinates)
- Filters for pedestrian class
- Generates train/val splits with proper directory structure
```

The conversion preserves critical metadata needed for height-based analysis while adapting to YOLO's training requirements.

#### Data Validation

```python
# scripts/data_checks.py
- Validates annotation completeness
- Checks for corrupted images
- Verifies class distribution
- Ensures height metadata integrity
```

### 5. End-to-End Pipeline

The project provides an automated pipeline for reproducible experiments:

```python
# scripts/run_pipeline.py
1. Training: Trains model with height-weighted data
2. Detection: Runs inference on validation set
3. Auditing: Performs fairness analysis and generates reports
```

This pipeline architecture enables:
- **Rapid experimentation**: Test different mitigation strategies quickly
- **Consistent evaluation**: Standardized metrics across all experiments
- **Result tracking**: Automated logging to CSV for longitudinal analysis

## Project Structure

```
fair-pedestrian-detection/
├── dataset/
│   └── bdd100k/
│       ├── images/              # Training and validation images
│       ├── labels/              # YOLO format annotations
│       ├── bdd100k_small.yaml   # Standard training config
│       └── bdd100k_height_weighted.yaml  # Fairness-aware config
├── scripts/
│   ├── train.py                 # Model training with height weighting
│   ├── detect.py                # Inference on validation set
│   ├── audit.py                 # General fairness auditing
│   ├── audit_height_bias.py     # Height-specific bias analysis
│   ├── convert_bdd100k_to_yolo.py  # Dataset format conversion
│   ├── data_checks.py           # Dataset validation utilities
│   ├── export_detections.py     # Export detection results with metadata
│   ├── enrich_predictions.py    # Add height group labels to predictions
│   └── run_pipeline.py          # End-to-end training and evaluation
├── models/                      # Trained model checkpoints
├── runs/                        # Experiment outputs
│   ├── train/                   # Training logs and weights
│   └── detect/
│       ├── predict/             # Detection results
│       └── audit/               # Fairness audit reports
├── logs/                        # Metrics tracking (CSV format)
├── requirements.txt             # Python dependencies
├── yolov8n.pt                   # Pretrained YOLOv8 weights
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, recommended for training)
- 10GB+ disk space for BDD100K dataset

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fair-pedestrian-detection.git
cd fair-pedestrian-detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download BDD100K dataset:

Visit [BDD100K website](https://bdd-data.berkeley.edu/) and download the detection dataset. Place images and annotations in `dataset/bdd100k/`.

5. Convert dataset to YOLO format:

```bash
python scripts/convert_bdd100k_to_yolo.py
```

## Usage

### Complete Pipeline

Run the full training, detection, and auditing workflow:

```bash
python scripts/run_pipeline.py
```

This executes:
1. Height-weighted model training
2. Inference on 500 sampled validation images
3. Bias auditing with fairness metrics computation

### Individual Components

#### Train a Model

Standard training:
```bash
python scripts/train.py
```

For height-weighted training, modify the `data` parameter in `train.py`:
```python
data="dataset/bdd100k/bdd100k_height_weighted.yaml"
```

#### Run Detection

```bash
python scripts/detect.py
```

Runs inference and saves visualizations to `runs/detect/predict/`.

#### Export Detection Metadata

```bash
python scripts/export_detections.py
```

Generates CSV file with per-instance detection data (coordinates, confidence, height).

#### Enrich Predictions

```bash
python scripts/enrich_predictions.py
```

Adds height group labels to detection CSV for analysis.

#### Audit Height Bias

```bash
python scripts/audit_height_bias.py
```

Computes height-stratified performance metrics and generates summary report.

## Evaluation Metrics

### Standard Object Detection Metrics

- **mAP@0.5**: Mean average precision at 0.5 IoU threshold
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Box Loss**: Localization accuracy (CIoU loss)
- **Classification Loss**: Confidence prediction error

### Fairness Metrics

#### Height Detection Rate (HDR)

Measures detection rate for each height group:
- **HDR_short**: Detection rate for pedestrians with height < 100px
- **HDR_tall**: Detection rate for pedestrians with height >= 100px

#### Height Detection Ratio

```
Ratio = HDR_short / HDR_tall
```

Quantifies relative performance disparity. Values closer to 1.0 indicate fairer performance.

#### Confidence Score Analysis

- **Mean confidence per group**: Average confidence scores for short vs. tall pedestrians
- **Confidence gap**: Difference in mean confidence between groups
- **Distribution analysis**: Histograms and statistical tests for confidence score distributions

### Height Group Definitions

Height groups are defined based on bounding box pixel height:
- **Short**: height < 100 pixels
  - Corresponds to children, distant pedestrians, or occluded individuals
  - Typically underrepresented in training data
- **Tall**: height >= 100 pixels
  - Corresponds to adults at typical distances
  - Majority class in most driving datasets

The 100-pixel threshold is empirically derived from the BDD100K dataset distribution and corresponds approximately to 140cm physical height in typical driving scenarios.

## Results

Results are automatically saved to organized directories:

### Training Results

```
runs/train/
├── weights/
│   ├── best.pt              # Best model checkpoint
│   └── last.pt              # Final epoch checkpoint
├── results.csv              # Per-epoch metrics
└── args.yaml                # Training configuration
```

### Detection Results

```
runs/detect/predict/
├── predictions.csv          # Per-instance detections
└── <image_name>.jpg         # Visualizations with bounding boxes
```

### Audit Results

```
runs/detect/audit/
├── enriched_predictions.csv      # Detections with height group labels
└── height_confidence_summary.csv # Group-wise statistics
```

### Metrics Logging

Training and evaluation metrics are logged to:

```
logs/metrics.csv
```

Contains columns: timestamp, run, mAP, HDR, box_loss, cls_loss

## Dataset

This project uses the [BDD100K](https://bdd-data.berkeley.edu/) dataset, a large-scale diverse driving video database with 100,000 images.

### Dataset Characteristics

- **Scale**: 100K images with bounding box annotations
- **Diversity**: Multiple weather conditions, times of day, and locations
- **Classes**: 10 object categories including pedestrian, car, truck, bus, etc.
- **Annotations**: Bounding boxes with class labels
- **Split**: 70K train / 10K validation / 20K test

### Why BDD100K?

BDD100K is ideal for studying height-based bias because:
1. **Real-world driving scenarios**: Captures authentic pedestrian appearances and distances
2. **Natural height variation**: Includes children, adults, and pedestrians at various distances
3. **Environmental diversity**: Multiple contexts prevent overfitting to specific conditions
4. **Scale**: Sufficient data for statistical significance in fairness analysis

## Technical Details

### Model Architecture

- **Base Model**: YOLOv8n (nano variant)
- **Input Size**: 320x320 pixels (optimized for speed)
- **Backbone**: CSPDarknet with PANet
- **Detection Head**: Anchor-free with distribution focal loss

### Training Hyperparameters

- **Optimizer**: SGD with momentum
- **Learning Rate**: 0.01 (initial)
- **Batch Size**: 8
- **Epochs**: Configurable (default: 1 for rapid prototyping)
- **Image Size**: 320x320
- **Classes**: Pedestrian only (class 0)

### Hardware Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional)
- **CPU**: Multi-core processor for data loading
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for dataset and results

## Acknowledgments

- **BDD100K Dataset**: UC Berkeley DeepDrive
- **YOLOv8 Framework**: Ultralytics
- **Fairness Research**: Various academic institutions studying algorithmic fairness in computer vision

## References

1. BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning (CVPR 2020)
2. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
3. Research on height-based bias in pedestrian detection systems
4. Fairness and bias mitigation in object detection literature
