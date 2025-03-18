# LEGO Bricks ML Vision - Single-class Training Pipeline Guide

This document provides a detailed technical overview of the single-class training pipeline implemented in `train_singleclass.py`. The script specializes in training individual detection models for either LEGO bricks or studs.

## Overview

The single-class training pipeline is a specialized version of the YOLO training system that focuses on detecting one type of LEGO element at a time. It supports two modes:
- Bricks detection (whole LEGO bricks)
- Studs detection (individual connection points)

## Key Components

### 1. Hardware Detection and Setup
```python
def detect_hardware():
```
- Automatically detects available GPU devices using PyTorch
- Provides graceful fallback to CPU with user confirmation
- Returns device configuration string for training

### 2. Dataset Management

#### Pre-training Verification
```python
def check_and_clean_directories(repo_root: Path, yes: bool = False):
```
- Verifies if cache or results directories contain existing data
- Prompts for cleanup if directories are not empty
- Supports automated cleanup with `--yes` flag
- Creates fresh directories if needed
- Ensures clean training environment

#### Dataset Extraction
```python
def extract_dataset(repo_root: Path, mode: str) -> Path:
```
- Extracts mode-specific dataset from:
  - `presentation/Datasets_Compress/bricks_dataset.zip` or
  - `presentation/Datasets_Compress/studs_dataset.zip`
- Creates organized workspace in `cache/datasets/{mode}`
- Uses Rich progress bars for visual feedback
- Returns path to extracted dataset

#### Dataset Structure Creation
```python
def create_dataset_structure(dataset_path: Path, train_ratio=0.7, val_ratio=0.2):
```
- Creates YOLO-compatible directory structure
- Splits data into train/validation/test sets
- Maintains image-label pair integrity
- Supports customizable split ratios
- Default split: 70% train, 20% validation, 10% test

### 3. YOLO Configuration

#### Dataset YAML Generation
```python
def create_dataset_yaml(dataset_path: Path, mode: str) -> Path:
```
- Creates YOLO dataset configuration file
- Single-class setup with mode-specific class name
- Includes:
  - Dataset paths
  - Class count (always 1)
  - Class name (either "bricks" or "studs")
  - Split directories

### 4. Training Pipeline

#### Model Training
```python
def train_model(yaml_path: Path, mode: str, device: str, epochs: int = 100, batch_size: int = 16):
```
- Initializes YOLOv8n model
- Mode-specific configuration
- Training parameters:
  - Epochs (default: 100)
  - Batch size (default: 16)
  - Early stopping (patience: 5)
  - Pretrained weights
- Creates mode-specific results directory

#### Results Management
```python
def zip_results(results_dir: Path, mode: str) -> Path:
```
- Archives training results with mode prefix
- Includes all training artifacts
- Saves outside repository for preservation
- Uses mode-specific naming convention

### 5. Command Line Interface (CLI) Usage

The script provides a comprehensive CLI with two main commands: `train` and `cleanup`.

#### Basic Usage

```bash
# Train a brick detection model
python train_singleclass.py train --mode bricks

# Train a stud detection model
python train_singleclass.py train --mode studs

# Clean up training artifacts
python train_singleclass.py cleanup
```

#### Training Options

1. **Mode Selection (Required)**
   ```bash
   python train_singleclass.py train --mode [bricks|studs]
   ```
   - Mandatory option specifying detection type
   - Determines dataset and class configuration

2. **Training Parameters**
   ```bash
   python train_singleclass.py train --mode bricks --epochs 150 --batch-size 32
   ```
   - `--epochs`: Training iterations (default: 100)
   - `--batch-size`: Batch size (default: 16)
   - `--train-ratio`: Training set proportion (default: 0.7)
   - `--val-ratio`: Validation set proportion (default: 0.2)

3. **Hardware Options**
   ```bash
   python train_singleclass.py train --mode studs --force-gpu
   ```
   - `--force-gpu`: Require GPU for training
   - Fails if GPU unavailable

4. **Automation**
   ```bash
   python train_singleclass.py train --mode bricks --yes
   ```
   - `--yes`: Skip all confirmations
   - Useful for automated workflows

#### Cleanup Options

1. **Full Cleanup**
   ```bash
   python train_singleclass.py cleanup --all
   ```
   - Removes all generated files
   - Includes cache and results

2. **Selective Cleanup**
   ```bash
   python train_singleclass.py cleanup --cache
   python train_singleclass.py cleanup --results
   ```
   - `--cache`: Clean dataset cache
   - `--results`: Clean training results

3. **Automated Cleanup**
   ```bash
   python train_singleclass.py cleanup --all --yes
   ```
   - Skip confirmation prompts
   - Use with caution

## Best Practices

1. **Mode Selection**
   - Choose mode based on detection needs
   - Verify dataset availability before training
   - Use appropriate model for each task

2. **Resource Management**
   - Clean workspace before new training sessions
   - Archive important results before cleanup
   - Monitor disk space during training

3. **Training Configuration**
   - Start with default parameters
   - Adjust batch size for available memory
   - Use early stopping for efficiency

4. **Results Handling**
   - Keep mode-specific results organized
   - Archive results immediately after training
   - Maintain clean working directories

## Error Handling

The pipeline includes comprehensive error handling for:
- Invalid mode selection
- Missing datasets
- Directory cleanup conflicts
- Training failures
- Hardware compatibility

## Logging and Progress

- Rich console formatting
- Mode-specific progress tracking
- Detailed error reporting
- Training metrics visualization