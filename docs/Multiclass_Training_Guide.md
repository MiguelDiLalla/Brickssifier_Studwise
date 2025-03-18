# LEGO Bricks ML Vision - Multiclass Training Pipeline Guide

This document provides a detailed technical overview of the multiclass training pipeline implemented in `train_multiclass.py`.

## Overview

The multiclass training pipeline is designed to train a YOLOv8 model for detecting and classifying different types of LEGO bricks in a single pass. The script handles the complete workflow from dataset preparation to model training and results archival.

## Key Components

### 1. Hardware Detection and Setup
```python
def detect_hardware():
```
- Automatically detects available GPU devices using PyTorch
- Provides graceful fallback to CPU with user confirmation
- Returns device configuration string for training

### 2. Dataset Management

#### Dataset Extraction
```python
def extract_dataset(repo_root: Path) -> Path:
```
- Extracts dataset from `presentation/Datasets_Compress/multiclass_dataset.zip`
- Creates clean workspace in `cache/datasets/multiclass`
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

#### Class Management
```python
def get_classes(dataset_path: Path) -> list:
```
- Extracts class definitions from metadata JSON
- Ensures consistent class ID ordering
- Supports dynamic class configurations

### 3. YOLO Configuration

#### Dataset YAML Generation
```python
def create_dataset_yaml(dataset_path: Path, classes: list) -> Path:
```
- Creates YOLO dataset configuration file
- Includes:
  - Dataset paths
  - Class count
  - Class names
  - Split directories

### 4. Training Pipeline

#### Pre-training Verification
```python
def check_and_clean_directories(repo_root: Path, yes: bool = False):
```
- Checks if cache or results directories have existing content
- Prompts user for cleanup confirmation if directories are not empty
- Can be automated with `--yes` flag
- Creates necessary directories if they don't exist
- Ensures clean workspace before training starts

#### Model Training
```python
def train_model(yaml_path: Path, device: str, epochs: int = 100, batch_size: int = 16):
```
- Initializes YOLOv8n model
- Configures training parameters:
  - Epochs (default: 100)
  - Batch size (default: 16)
  - Early stopping (patience: 5)
  - Pretrained weights
- Saves results to organized directory structure

#### Results Management
```python
def zip_results(results_dir: Path) -> Path:
```
- Archives training results
- Includes:
  - Model weights
  - Training logs
  - Performance metrics
- Saves outside repository for preservation

### 5. Command Line Interface (CLI) Usage

The `train_multiclass.py` script provides a comprehensive command-line interface designed for flexibility and ease of use. The CLI is built using the Click framework, offering robust argument parsing and help documentation.

#### Basic Usage

The CLI has two main commands:
```bash
# Training command
python train_multiclass.py train

# Cleanup command
python train_multiclass.py clean
```

#### Detailed Command Reference

##### Train Command
```bash
python train_multiclass.py train [OPTIONS]
```

###### Training Options Explained

1. **Epochs Control**
   ```bash
   python train_multiclass.py train --epochs 150
   ```
   - `--epochs`: Number of training epochs
   - Default: 100
   - Range: 1-1000
   - Higher values may improve accuracy but increase training time
   - Early stopping will prevent overfitting

2. **Batch Size Management**
   ```bash
   python train_multiclass.py train --batch-size 32
   ```
   - `--batch-size`: Images processed per training step
   - Default: 16
   - Common values: 8, 16, 32, 64
   - Larger batches:
     - Faster training
     - Require more GPU memory
     - May affect model convergence

3. **Dataset Split Configuration**
   ```bash
   python train_multiclass.py train --train-ratio 0.8 --val-ratio 0.15
   ```
   - `--train-ratio`: Proportion for training set
     - Default: 0.7 (70%)
     - Range: 0.1-0.9
   - `--val-ratio`: Proportion for validation set
     - Default: 0.2 (20%)
     - Range: 0.1-0.4
   - Test set is automatically calculated (1 - train - val)
   - Recommendations:
     - Small datasets: 60/20/20 split
     - Large datasets: 80/10/10 split

4. **Hardware Control**
   ```bash
   python train_multiclass.py train --force-gpu
   ```
   - `--force-gpu`: Require GPU for training
     - Fails if no GPU available
     - Prevents accidental CPU training
   
5. **Automation Options**
   ```bash
   python train_multiclass.py train --yes
   ```
   - `--yes`: Skip all confirmation prompts
   - Useful for automated scripts
   - Exercise caution when using

###### Example Training Commands

1. Quick Test Run:
   ```bash
   python train_multiclass.py train --epochs 10 --batch-size 8
   ```

2. Production Training:
   ```bash
   python train_multiclass.py train --epochs 200 --batch-size 32 --force-gpu
   ```

3. Custom Dataset Split:
   ```bash
   python train_multiclass.py train --train-ratio 0.8 --val-ratio 0.1
   ```

4. Automated Pipeline:
   ```bash
   python train_multiclass.py train --epochs 100 --force-gpu --yes
   ```

##### Cleanup Command
```bash
python train_multiclass.py cleanup [OPTIONS]
```

###### Cleanup Options Explained

1. **Complete Cleanup**
   ```bash
   python train_multiclass.py cleanup --all
   ```
   - `--all`: Removes all generated files
   - Deletes:
     - Cache directory
     - Results directory
     - Temporary files
     - Training artifacts

2. **Selective Cleanup**
   ```bash
   python train_multiclass.py cleanup --cache
   python train_multiclass.py cleanup --results
   ```
   - `--cache`: Clean only cache directory
     - Removes extracted datasets
     - Preserves training results
   - `--results`: Clean only results directory
     - Removes training outputs
     - Preserves cached datasets

3. **Automated Cleanup**
   ```bash
   python train_multiclass.py cleanup --all --yes
   ```
   - `--yes`: Skip confirmation prompts
   - Use with caution to prevent data loss

###### Example Cleanup Commands

1. Safe Cleanup (with confirmation):
   ```bash
   python train_multiclass.py cleanup --cache
   ```

2. Full Reset:
   ```bash
   python train_multiclass.py cleanup --all
   ```

3. Automated Pipeline Cleanup:
   ```bash
   python train_multiclass.py cleanup --results --yes
   ```

#### Environment Variables

The CLI also respects several environment variables:

- `CUDA_VISIBLE_DEVICES`: Control GPU visibility
- `PYTORCH_CUDA_ALLOC_CONF`: Configure CUDA memory allocation
- `YOLO_VERBOSE`: Control YOLOv8 logging level

#### Error Handling

The CLI implements comprehensive error handling:

1. **Input Validation**
   - Checks parameter ranges
   - Validates file paths
   - Verifies hardware compatibility

2. **Runtime Protection**
   - Prevents duplicate training sessions
   - Handles interrupt signals gracefully
   - Saves partial progress on failure

3. **Recovery Options**
   - Automatic backup of important files
   - Resume capability for interrupted training
   - Cleanup tools for failed sessions

#### Logging and Output

The CLI provides rich feedback through multiple channels:

1. **Console Output**
   - Color-coded status messages
   - Progress bars for long operations
   - Real-time training metrics

2. **Log Files**
   - Detailed training logs
   - Error tracebacks
   - Performance metrics

3. **Results**
   - Training plots
   - Model checkpoints
   - Validation results

## Best Practices

1. **Hardware Usage**
   - Always prefer GPU training when available
   - Confirm CPU usage to prevent unexpected long training times

2. **Dataset Management**
   - Maintain proper image-label pair relationships
   - Use appropriate split ratios for dataset size
   - Verify class balance in splits

3. **Training Configuration**
   - Start with default parameters
   - Adjust batch size based on available memory
   - Monitor early stopping for optimal results

4. **Results Management**
   - Archive results immediately after training
   - Keep original zip archives for reference
   - Clean cache regularly to manage disk space

## Error Handling

The pipeline includes robust error handling for:
- Missing dataset files
- Invalid dataset structure
- Hardware compatibility issues
- Training failures
- File system operations

## Logging and Progress Tracking

- Uses Rich for formatted console output
- Provides progress bars for long operations
- Includes emoji indicators for status
- Maintains detailed logging for debugging