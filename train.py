"""
LEGO Bricks ML Vision - Training Pipeline

This module provides a complete ML training pipeline for LEGO Bricks and Studs detection
models using YOLOv8. It handles dataset preparation, training configuration, model training
and results management.

Key features:
  - Automatic hardware detection (CUDA/MPS/CPU)
  - Dataset extraction and validation 
  - Train/validation/test dataset splitting
  - Data augmentation via Albumentations
  - Model selection and training
  - Results compression and sharing
  - Detailed logging with emoji markers

Usage examples:
  - Use CLI interface: python lego_cli.py train --mode bricks
  - Direct API: from train import train_model, setup_logging

Author: Miguel DiLalla
"""

import os
import logging
import torch
import shutil
import zipfile
import json
import random
import datetime
import subprocess
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from IPython.display import FileLink, display
from pprint import pprint
import pandas as pd

# =============================================================================
# Logging Setup
# =============================================================================

class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emoji markers to log messages based on level."""
    def format(self, record):
        base_msg = super().format(record)
        if record.levelno >= logging.ERROR:
            emoji = "❌"
        elif record.levelno >= logging.WARNING:
            emoji = "⚠️"
        elif record.levelno >= logging.INFO:
            emoji = "✅"
        else:
            emoji = "💬"
        return f"{base_msg} {emoji}"

# Define a simple config loader function
def load_config():
    """
    Simple configuration loader for the notebook demo.
    
    Returns:
        dict: Configuration dictionary with training parameters
        
    Note:
        This function could be moved to train.py for better organization.
    """
    return {
        "training": {
            "batch_size": 16,
            "epochs": 50,
            "device": "auto",
            "use_pretrained": True
        }
    }

def setup_logging():
    """
    Configures structured logging for train.py with emoji markers.
    
    Sets up both console and file handlers with consistent formatting.
    Creates 'logs' directory if it doesn't exist.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_session.log")
    
    formatter = EmojiFormatter("%(asctime)s - %(levelname)s - %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    
    # Clear existing handlers to enforce our configuration
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler]
    )
    
    logging.info("Logging initialized for training pipeline")

# =============================================================================
# Environment Setup
# =============================================================================

def get_repo_root() -> Path:
    """
    Auto-detects the repository root directory.
    
    Returns:
        Path: The repository root directory path.
        
    Notes:
        - Searches for .git directory moving up from current location
        - Falls back to current working directory if not found
    """
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:  # Avoid infinite loops
        if (current_dir / ".git").exists():
            return current_dir
        current_dir = current_dir.parent  # Move up one level

    # If no .git found, return the current execution directory
    return Path.cwd()

def detect_hardware():
    """
    Detects available hardware for training.
    
    Returns:
        str: Device specification for PyTorch (e.g., "0,1" for multiple GPUs, "cpu", or "mps")
        
    Notes:
        - Checks for CUDA-capable GPUs and returns all available devices
        - Falls back to Apple MPS if available
        - Defaults to CPU with warning if no accelerators found
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = ",".join(str(i) for i in range(num_gpus))
        logging.info(f"Detected {num_gpus} GPU(s): {device}")
        return device  # Returning the actual device list like "0,1"
    
    elif torch.backends.mps.is_available():
        logging.info("Detected Apple MPS device.")
        return "mps"
    
    logging.warning("No GPU or MPS device detected. Falling back to CPU.")
    return "cpu"

def setup_execution_structure(repo_root):
    """
    Ensures all necessary cache directories exist before execution.
    
    Creates standardized folder structure for:
      - Cache (datasets, models, logs)
      - Results
      
    Returns:
        None
    """
    
    required_dirs = [
        repo_root / "cache" / "datasets",
        repo_root / "cache" / "models",
        repo_root / "cache" / "logs",
        repo_root / "results"  # ✅ FIXED: Ensuring results go to the right place
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    logging.info("✅ Execution structure initialized.")

def cleanup_training_sessions(repo_root):
    """
    Cleans up temporary directories after training completion.
    
    Removes:
      - cache/
      - results/
      
    Returns:
        None
      
    Notes:
        - Preserves original dataset files
        - Logs successful and failed cleanup operations
    """
    
    folders = ["cache", "results"]
    for folder in folders:
        folder_path = os.path.join(repo_root, folder)
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    shutil.rmtree(os.path.join(root, dir))
            logging.info(f"✅ Emptied: {folder_path}")
        else:
            logging.warning(f"❌ Not found: {folder_path}. Creating it now.")
            os.makedirs(folder_path, exist_ok=True)
            logging.info(f"✅ Created: {folder_path}")

def export_logs(log_name="train_session"):
    """
    Exports logs and hardware details in JSON format.
    
    Args:
        log_name (str): Base name for the log file (without extension)
        
    Returns:
        str: Path to the exported JSON file
        
    Notes:
        - Includes hardware details (GPU, CUDA version, etc.)
        - Preserves all log entries in structured format
    """
    log_path = os.path.join("logs", f"{log_name}.log")
    export_path = log_path.replace(".log", ".json")
    
    hardware_info = {
        "python_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "num_gpus": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(log_path, "r") as f:
        log_entries = [line.strip() for line in f.readlines()]
    
    session_data = {
        "hardware_info": hardware_info,
        "logs": log_entries
    }
    
    with open(export_path, "w") as f:
        json.dump(session_data, f, indent=4)
    
    logging.info(f"✅ Logs exported to {export_path}")
    return export_path

# =============================================================================
# Dataset Preparation
# =============================================================================

def unzip_dataset(mode, force_extract=False):
    """
    Extracts the dataset from the repository's compressed files.

    Args:
        mode (str): 'bricks' or 'studs'.
        force_extract (bool): If True, forces re-extraction even if dataset exists.
        
    Returns:
        str: Path to the extracted dataset
        
    Notes:
        - Checks if dataset was previously extracted to avoid redundant work
        - Creates necessary directories automatically
    """
    repo_root = get_repo_root()
    dataset_compressed_dir = os.path.join(repo_root, "presentation/Datasets_Compress")
    dataset_dir = os.path.join(repo_root, "cache/datasets")
    
    dataset_filename = "bricks_dataset.zip" if mode == "bricks" else "studs_dataset.zip"
    dataset_path = os.path.join(dataset_compressed_dir, dataset_filename)
    extract_path = os.path.join(dataset_dir, mode)

    os.makedirs(dataset_dir, exist_ok=True)

    if not force_extract and os.path.exists(extract_path):
        logging.info(f"Dataset already extracted at {extract_path}. Skipping extraction.")
        return extract_path

    logging.info(f"Extracting {dataset_filename}...")
    shutil.unpack_archive(dataset_path, extract_path)
    logging.info(f"Dataset extracted to: {extract_path}")

    return extract_path

def validate_dataset(mode):
    """
    Validates and cleans the unzipped dataset by:
    1. Detecting images and labels folders
    2. Removing corrupted images and invalid label files
    3. Ensuring 1:1 image-label pairing
    4. Logging all validation steps

    Args:
        mode (str): 'bricks' or 'studs'.

    Returns:
        tuple: (images_path, labels_path) paths to valid dataset folders
    """
    logging.info(f"Starting dataset validation for {mode} mode")
    
    # Get paths
    repo_root = get_repo_root()
    dataset_path = os.path.join(repo_root, "cache/datasets", mode)
    
    # Find subfolders
    subfolders = [f for f in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, f))]
    
    if len(subfolders) != 2:
        logging.error(f"Expected 2 subfolders, found {len(subfolders)}")
        raise ValueError("Invalid dataset structure")

    # Detect images and labels folders
    images_path = None
    labels_path = None
    
    for folder in subfolders:
        folder_path = os.path.join(dataset_path, folder)
        sample_files = os.listdir(folder_path)[:5]  # Check first 5 files
        
        # Check if folder contains images
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in sample_files):
            images_path = folder_path
        # Check if folder contains text files
        elif any(f.lower().endswith('.txt') for f in sample_files):
            labels_path = folder_path
    
    if not images_path or not labels_path:
        logging.error("Could not identify images and labels folders")
        raise ValueError("Invalid dataset structure")
        
    logging.info(f"Detected - Images: {images_path}, Labels: {labels_path}")

    # Validate images
    valid_images = []
    for img in os.listdir(images_path):
        if not img.lower().endswith(('.jpg')):
            continue
            
        img_path = os.path.join(images_path, img)
        try:
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Removing corrupted image: {img}")
                os.remove(img_path)
            else:
                valid_images.append(os.path.splitext(img)[0])
        except Exception as e:
            logging.warning(f"Error processing image {img}: {e}")
            os.remove(img_path)

    # Validate labels
    valid_labels = []
    for label in os.listdir(labels_path):
        if not label.endswith('.txt'):
            continue
            
        label_path = os.path.join(labels_path, label)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            # Check YOLO format: each line should be "class x y w h"
            format_errors = []
            for line_num, line in enumerate(lines, 1):
                if line.strip():  # Skip empty lines
                    try:
                        values = line.split()
                        if len(values) != 5:
                            format_errors.append(f"Line {line_num}: Expected 5 values, got {len(values)}")
                        elif not all(0 <= float(x) <= 1 for x in values[1:]):
                            format_errors.append(f"Line {line_num}: Values must be between 0 and 1")
                    except ValueError:
                        format_errors.append(f"Line {line_num}: Invalid number format")
                
            if not lines or format_errors:
                if not lines:
                    logging.warning(f"File was empty: {label}")
                logging.warning(f"Removing invalid label file {label}:")
                for error in format_errors:
                    logging.warning(f"  {error}")
                os.remove(label_path)
            else:
                valid_labels.append(Path(label).stem)
                
        except Exception as e:
            logging.warning(f"Error processing label {label}: {e}")
            os.remove(label_path)


    # Ensure 1:1 pairing
    valid_pairs = set(valid_images) & set(valid_labels)
    orphaned_images = set(valid_images) - valid_pairs
    orphaned_labels = set(valid_labels) - valid_pairs
    
    # Remove orphaned files
    for orphan in orphaned_images:
        os.remove(os.path.join(images_path, f"{orphan}.jpg"))
    for orphan in orphaned_labels:
        os.remove(os.path.join(labels_path, f"{orphan}.txt"))
        
    # Log statistics
    logging.info(f"Dataset validation complete:")
    logging.info(f"- Valid image-label pairs: {len(valid_pairs)}")
    logging.info(f"- Removed images: {len(orphaned_images)}")
    logging.info(f"- Removed labels: {len(orphaned_labels)}")
    
    if not valid_pairs:
        logging.error("No valid image-label pairs found")
        raise ValueError("Empty dataset after validation")
        
    return images_path, labels_path
        


def create_dataset_structure(mode, repo_root):
    """
    Creates necessary dataset directories for YOLO.
    
    Args:
        mode (str): 'bricks' or 'studs', defining dataset name
        
    Returns:
        Path: Path to the created dataset structure root
        
    Notes:
        - Creates standard YOLO folder structure with train/val/test splits
        - Separate directories for images and labels
    """
    output_dir = repo_root / "cache" / "split" / f"{mode}"
    yolo_dirs = [
        output_dir / "dataset" / "images" / "train",
        output_dir / "dataset" / "images" / "val",
        output_dir / "dataset" / "images" / "test",
        output_dir / "dataset" / "labels" / "train",
        output_dir / "dataset" / "labels" / "val",
        output_dir / "dataset" / "labels" / "test"
    ]
    
    for yolo_dir in yolo_dirs:
        yolo_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"✅ Dataset structure created at {output_dir}")
    return output_dir

def augment_data(dataset_path, augmentations=2):
    """Enhanced data augmentation with robust error handling."""
    
    # Setup paths and validation
    train_images_path = os.path.join(dataset_path, "dataset/images/train")
    train_labels_path = os.path.join(dataset_path, "dataset/labels/train")
    
    # Verify disk space
    free_space = shutil.disk_usage(train_images_path).free
    required_space = sum(os.path.getsize(os.path.join(train_images_path, f)) 
                        for f in os.listdir(train_images_path))
    if free_space < required_space * augmentations * 1.5:
        raise RuntimeError("Insufficient disk space for augmentation")
    
    # Configure augmentation pipeline with bbox support
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.GaussianBlur(p=0.2),
        A.ColorJitter(p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    stats = {"processed": 0, "augmented": 0, "failed": 0}
    
    for img_file in os.listdir(train_images_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        try:
            # Load and validate image
            img_path = os.path.join(train_images_path, img_file)
            label_path = os.path.join(train_labels_path, 
                                    os.path.splitext(img_file)[0] + '.txt')
            
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_file}")
            
            # Load and validate labels
            bboxes = []
            class_labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        cls_id, x, y, w, h = map(float, line.strip().split())
                        bboxes.append([x, y, w, h])
                        class_labels.append(cls_id)
            
            # Perform augmentations
            for i in range(augmentations):
                try:
                    transformed = transform(image=image, bboxes=bboxes, 
                                         class_labels=class_labels)
                    
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    # Save augmented image
                    aug_name = f"{os.path.splitext(img_file)[0]}_aug{i}{os.path.splitext(img_file)[1]}"
                    cv2.imwrite(os.path.join(train_images_path, aug_name), aug_image)
                    
                    # Save augmented labels
                    if bboxes:
                        label_name = f"{os.path.splitext(img_file)[0]}_aug{i}.txt"
                        with open(os.path.join(train_labels_path, label_name), 'w') as f:
                            for bbox, cls_id in zip(aug_bboxes, aug_labels):
                                f.write(f"{int(cls_id)} {' '.join(map(str, bbox))}\n")
                    
                    stats["augmented"] += 1
                    
                except Exception as e:
                    logging.error(f"Failed augmentation {i} for {img_file}: {str(e)}")
                    stats["failed"] += 1
                    continue
            
            stats["processed"] += 1
            
            # Log progress
            if stats["processed"] % 10 == 0:
                logging.info(f"Progress: {stats['processed']} images processed, "
                           f"{stats['augmented']} augmentations created, "
                           f"{stats['failed']} failures")
                
        except Exception as e:
            logging.error(f"Failed to process {img_file}: {str(e)}")
            stats["failed"] += 1
            continue
    
    # Final report
    logging.info("Augmentation Complete:")
    logging.info(f"  Images processed: {stats['processed']}")
    logging.info(f"  Augmentations created: {stats['augmented']}")
    logging.info(f"  Failed operations: {stats['failed']}")

def dataset_split(mode, repo_root):
    """
    Splits dataset into train (70%), val (20%), test (10%) and creates YOLO-compatible structure.
    
    Args:
        mode (str): 'bricks' or 'studs', defining dataset to split
        
    Returns:
        str: Path to the YOLO dataset directory
    """
    logging.info(f"Starting dataset split for {mode} mode")
    
    dataset_path = os.path.join(repo_root, "cache/datasets", mode)
    output_dir = os.path.join(repo_root, "cache/split", f"{mode}")

    subfolders = [f for f in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, f))]

    # Detect images and labels folders
    images_path = None
    labels_path = None
    
    for folder in subfolders:
        folder_path = os.path.join(dataset_path, folder)
        sample_files = os.listdir(folder_path)[:5]  # Check first 5 files
        
        # Check if folder contains images
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in sample_files):
            images_path = folder_path
        # Check if folder contains text files
        elif any(f.lower().endswith('.txt') for f in sample_files):
            labels_path = folder_path
    
    if not images_path or not labels_path:
        logging.error("Could not identify images and labels folders")
        raise ValueError("Invalid dataset structure")
        
    logging.info(f"Detected - Images: {images_path}, Labels: {labels_path}")
    
    logging.info(f"Source dataset: {dataset_path}")
    logging.info(f"Output directory: {output_dir}")

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
    logging.info(f"Found {len(image_files)} images in dataset")
    
    random.shuffle(image_files)
    logging.info("Files randomly shuffled for unbiased split")

    num_train = int(len(image_files) * 0.7)
    num_val = int(len(image_files) * 0.2)

    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]
    
    logging.info(f"Split sizes: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

    # Ensure destination folders exist
    def ensure_folder_exists(path):
        os.makedirs(path, exist_ok=True)
        logging.debug(f"Ensured directory exists: {path}")

    logging.info("Creating YOLO directory structure")
    ensure_folder_exists(os.path.join(output_dir, "dataset/images/train"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/images/val"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/images/test"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/labels/train"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/labels/val"))
    ensure_folder_exists(os.path.join(output_dir, "dataset/labels/test"))

    def move_files(files, img_dst, lbl_dst):
        copied = 0
        missing_labels = 0
        for f in files:
            img_src = os.path.join(images_path, f)
            lbl_src = os.path.join(labels_path, f.replace(".jpg", ".txt"))
            
            # Copy image
            shutil.copy(img_src, os.path.join(output_dir, img_dst, f))
            
            # Check if label exists before copying
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(output_dir, lbl_dst, f.replace(".jpg", ".txt")))
                copied += 1
            else:
                missing_labels += 1
                logging.warning(f"Missing label for image {f}")
        
        logging.info(f"Copied {copied} image-label pairs to {img_dst}")
        if missing_labels > 0:
            logging.warning(f"Found {missing_labels} images without labels in {img_dst}")

    logging.info("Copying files to train split...")
    move_files(train_files, "dataset/images/train", "dataset/labels/train")
    
    logging.info("Copying files to validation split...")
    move_files(val_files, "dataset/images/val", "dataset/labels/val")
    
    logging.info("Copying files to test split...")
    move_files(test_files, "dataset/images/test", "dataset/labels/test")

    dataset_yaml = {
        "path": output_dir,
        "train": "dataset/images/train",
        "val": "dataset/images/val", 
        "test": "dataset/images/test",
        "nc": 1,
        "names": {
            0: "brick" if mode == "bricks" else "stud"
        }
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    # empty the original dataset folder and log the completion
    shutil.rmtree(dataset_path)
    logging.info(f"✅ Original dataset folder emptied: {dataset_path}")

    logging.info(f"✅ Dataset split completed. Updated dataset.yaml at {yaml_path}")
    logging.info(f"Dataset ready for training with class: {dataset_yaml['names'][0]}")
    return output_dir

def split_dataset(mode, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        mode (str): 'bricks' or 'studs'.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.

    Returns:
        dict: Paths to the training, validation, and test sets.
    """
    repo_root = get_repo_root()
    dataset_path = os.path.join(repo_root, "cache/datasets", mode)
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
    label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(".txt")])

    # Ensure the dataset is valid
    if len(image_files) != len(label_files):
        logging.error("Mismatch between number of images and labels.")
        raise ValueError("Image-label count mismatch. Ensure every image has a corresponding label.")

    # Shuffle and split the dataset
    combined = list(zip(image_files, label_files))
    random.shuffle(combined)
    image_files[:], label_files[:] = zip(*combined)

    train_split = int(train_ratio * len(image_files))
    val_split = int((train_ratio + val_ratio) * len(image_files))

    train_images, val_images, test_images = image_files[:train_split], image_files[train_split:val_split], image_files[val_split:]
    train_labels, val_labels, test_labels = label_files[:train_split], label_files[train_split:val_split], label_files[val_split:]

    # Create directories for splits
    split_paths = {
        "train": {"images": os.path.join(images_path, "train"), "labels": os.path.join(labels_path, "train")},
        "val": {"images": os.path.join(images_path, "val"), "labels": os.path.join(labels_path, "val")},
        "test": {"images": os.path.join(images_path, "test"), "labels": os.path.join(labels_path, "test")}
    }

    for split in split_paths.values():
        os.makedirs(split["images"], exist_ok=True)
        os.makedirs(split["labels"], exist_ok=True)

    # Move files to respective directories
    for img, lbl in zip(train_images, train_labels):
        shutil.move(os.path.join(images_path, img), os.path.join(split_paths["train"]["images"], img))
        shutil.move(os.path.join(labels_path, lbl), os.path.join(split_paths["train"]["labels"], lbl))

    for img, lbl in zip(val_images, val_labels):
        shutil.move(os.path.join(images_path, img), os.path.join(split_paths["val"]["images"], img))
        shutil.move(os.path.join(labels_path, lbl), os.path.join(split_paths["val"]["labels"], lbl))

    for img, lbl in zip(test_images, test_labels):
        shutil.move(os.path.join(images_path, img), os.path.join(split_paths["test"]["images"], img))
        shutil.move(os.path.join(labels_path, lbl), os.path.join(split_paths["test"]["labels"], lbl))

    logging.info(f"✅ Dataset split into train, val, and test sets for mode: {mode}")
    return split_paths

def display_dataset_info(mode):
    try:
        result = validate_dataset(mode)
        if result is None:
            # Fallback paths if validate_dataset returns None
            images_path = f"datasets/{mode}/images"
            labels_path = f"datasets/{mode}/labels"
            print(f"Warning: validate_dataset returned None. Using fallback paths.")
        else:
            images_path, labels_path = result
            
        # Rest of your existing function
        
        return images_path, labels_path
    except Exception as e:
        print(f"Error processing dataset info: {e}")
        # Return fallback values to avoid unpacking errors
        return f"datasets/{mode}/images", f"datasets/{mode}/labels"

# =============================================================================
# Training Functions
# =============================================================================

def select_model(mode, use_pretrained=False):
    """
    Selects a pre-trained model from the repository or defaults to YOLOv8n.

    Args:
        mode (str): 'bricks' or 'studs'.
        use_pretrained (bool): If True, selects a LEGO-trained model, else defaults to YOLOv8n.

    Returns:
        str: Path to the selected model checkpoint.
        
    Raises:
        FileNotFoundError: If requested model file is not found
        
    Notes:
        - Falls back to YOLOv8n if use_pretrained is False
    """
    repo_root = get_repo_root()
    
    if not use_pretrained:
        logging.info("✅ Using default YOLOv8n model.")
        return "yolov8n.pt"
    
    model_dir = os.path.join(repo_root, "presentation/Models_DEMO")
    model_filename = "Brick_Model_best20250123_192838t.pt" if mode == "bricks" else "Stud_Model_best20250124_170824.pt"
    model_path = os.path.join(model_dir, model_filename)
    
    if os.path.exists(model_path):
        logging.info(f"✅ Model selected: {model_path}")
        return model_path
    else:
        logging.error(f"❌ Model not found at {model_path}")
        raise FileNotFoundError(f"Required model file is missing: {model_path}")

def save_model(model, output_dir, model_name="trained_model.pt"):
    """
    Saves the trained model to the specified directory.
    
    Args:
        model: YOLO model object to save
        output_dir (str): Directory to save the model
        model_name (str): Filename for the saved model
        
    Returns:
        str: Full path to the saved model file
    """
    model_save_path = os.path.join(output_dir, model_name)
    model.save(model_save_path)
    logging.info(f"✅ Model saved to: {model_save_path}")
    return model_save_path

def train_model(dataset_path, model_path, device, epochs, batch_size, repo_root):
    """
    Trains the YOLOv8 model with robust error handling and fallback mechanisms.
    
    Args:
        dataset_path (str): Path to the dataset directory
        model_path (str): Path to the model file or pre-trained model name
        device (str): Training device specification (e.g., "0" or "cpu")
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        repo_root (str): Path to the repository root directory
        
    Returns:
        str: Path to the directory containing training results
        
    Notes:
        - Attempts CLI method first then falls back to Python API if needed
        - Validates paths before execution to provide helpful error messages
        - Results are saved with timestamped directory names
    """
    logging.info(f"🚀 Starting training with model: {model_path}")
    
    # Validate inputs
    dataset_yaml_path = os.path.join(dataset_path, "dataset.yaml")

    # retrieve mode from yaml file
    with open(dataset_yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
        mode = dataset_config["names"][0]
        if mode == "lego_brick":
            mode = "bricks"
        elif mode == "lego_stud":
            mode = "studs"
        # log mode detected
        logging.info(f"Mode detected: {mode} 🐯")

    if not os.path.exists(dataset_yaml_path):
        logging.error(f"❌ Dataset YAML not found: {dataset_yaml_path}")
        raise FileNotFoundError(f"Dataset YAML file not found: {dataset_yaml_path}")
    
    # Verify dataset YAML content
    try:
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
            logging.info(f"Dataset config: {dataset_config}")
            
            # Check required keys
            required_keys = ['path', 'train', 'val', 'test', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in dataset_config]
            if missing_keys:
                logging.error(f"❌ Dataset YAML missing required keys: {missing_keys}")
                raise ValueError(f"Dataset YAML missing required keys: {missing_keys}")
            
            # Validate paths in the YAML
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(dataset_path, dataset_config[split])
                if not os.path.exists(split_path):
                    logging.warning(f"⚠️ {split} path doesn't exist: {split_path}")
    except Exception as e:
        logging.error(f"❌ Failed to validate dataset YAML: {e}")
        raise ValueError(f"Invalid dataset YAML: {e}")
    
    # Standard model presets don't need file existence check
    if not model_path.startswith("yolov8") and not os.path.exists(model_path):
        logging.error(f"❌ Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize model to verify it loads correctly
    try:
        model = YOLO(model_path)
        logging.info(f"✅ Model initialized successfully: {model.task}, {model.names}")
    except Exception as e:
        logging.error(f"❌ Failed to initialize model: {e}")
        raise RuntimeError(f"Failed to initialize YOLO model: {e}")
    
    # Setup training outputs
    training_name = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(repo_root, f"results/{mode}")
    os.makedirs(results_dir, exist_ok=True)
    
    logging.info(f"Project path: {results_dir} ✅")
    
    # Try Python API first as it's more reliable across platforms
    try:
        logging.info("Starting training through Python API...")
        results = model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            batch=batch_size,
            device=device,
            project=results_dir,
            name=training_name,
            patience=5,
            single_cls=True,
            exist_ok=True,
            verbose=True # Use the detected mode as class name
        )
        logging.info("✅ Training completed successfully via Python API.")
        return os.path.join(results_dir, training_name)
    except Exception as api_error:
        logging.warning(f"⚠️ Python API training failed: {api_error}. Attempting CLI method...")
        
        # Fall back to CLI method
        try:
            # Use normalized paths for CLI
            norm_model_path = os.path.normpath(model_path)
            norm_dataset_path = os.path.normpath(dataset_yaml_path)
            norm_results_dir = os.path.normpath(results_dir)
            
            command = [
                "yolo", 
                "task=detect",
                "train",
                f"model={norm_model_path}",
                f"data={norm_dataset_path}",
                f"epochs={epochs}",
                f"batch={batch_size}",
                f"device={device}",
                f"project={norm_results_dir}",
                f"name={training_name}",
                "patience=5",
                "single_cls=True",
                "exist_ok=True"
            ]
            
            logging.info(f"Executing command: {' '.join(command)}")
            
            # Capture and log both stdout and stderr
            process = subprocess.run(
                command, 
                check=False,
                text=True,
                capture_output=True
            )
            
            if process.returncode != 0:
                logging.error(f"❌ CLI command failed with exit code {process.returncode}")
                logging.error(f"Command output: {process.stdout}")
                logging.error(f"Command error: {process.stderr}")
                raise RuntimeError(f"CLI training failed with exit code {process.returncode}. See logs for details.")
                
            logging.info("✅ Training completed successfully via CLI.")
            return os.path.join(results_dir, training_name)
            
        except Exception as cli_error:
            logging.error(f"❌ Training failed via both Python API and CLI: {cli_error}")
            raise RuntimeError(f"Training failed: Original error - {api_error}, CLI error - {cli_error}")

# =============================================================================
# Results Management
# =============================================================================

def zip_and_download_results(results_dir=None, output_filename=None):
    """
    Compresses the results directory into a ZIP file and provides a download link.

    Args:
        results_dir (str, optional): The path to the results folder.
        output_filename (str, optional): Name of the output zip file.
        
    Returns:
        None
    """
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "results")
    if output_filename is None:
        output_filename = os.path.join(os.getcwd(), "..", "training_results.zip")

    # Convert Path object to string if needed
    if hasattr(output_filename, '__fspath__'):
        output_filename = str(output_filename)

    # First copy the logs folder to the results folder
    log_src = os.path.join(os.getcwd(), "logs")
    log_dst = os.path.join(results_dir, "logs")
    shutil.copytree(log_src, log_dst, dirs_exist_ok=True)   

    if not os.path.exists(results_dir):
        logging.error("❌ No results folder found.")
        return

    # Create ZIP file using string paths
    base_filename = output_filename.replace(".zip", "")
    zip_path = shutil.make_archive(base_filename, 'zip', results_dir)
    
    logging.info(f"✅ Training results compressed: {zip_path}")

    # Provide a direct download link
    display(FileLink(zip_path))

from rich.table import Table
from rich.console import Console
from rich.style import Style
import matplotlib.gridspec as gridspec
import numpy as np

def display_last_training_session(session_dir):
    """
    Displays all files from the specified training session directory with organized grids
    and rich formatting.
    """
    if not os.path.exists(session_dir):
        logging.error(f"Results directory not found: {session_dir}")
        return

    logging.info(f"Displaying training session: {session_dir}")
    files = sorted(os.listdir(session_dir))
    
    # Group files by type
    plot_files = [f for f in files if f.lower().endswith(('.jpg', '.png')) and 'batch' not in f]
    batch_files = [f for f in files if 'batch' in f.lower() and f.lower().endswith(('.jpg', '.png'))]
    csv_files = [f for f in files if f.lower().endswith('.csv')]
    text_files = [f for f in files if f.lower().endswith('.txt')]
    yaml_files = [f for f in files if f.lower().endswith('.yaml')]

    # Display plots in a grid
    if plot_files:
        n_plots = len(plot_files)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 5*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        
        for idx, file in enumerate(plot_files):
            file_path = os.path.join(session_dir, file)
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax = fig.add_subplot(gs[idx//n_cols, idx%n_cols])
                ax.imshow(image)
                ax.set_title(file)
                ax.axis('off')
        
        plt.tight_layout()
        display(fig)
        plt.close()

    # Display batch images in a grid
    if batch_files:
        n_batches = len(batch_files)
        n_cols = min(4, n_batches)
        n_rows = (n_batches + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 4*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        
        for idx, file in enumerate(batch_files):
            file_path = os.path.join(session_dir, file)
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax = fig.add_subplot(gs[idx//n_cols, idx%n_cols])
                ax.imshow(image)
                ax.set_title(f"Batch {idx+1}")
                ax.axis('off')
        
        plt.suptitle("Training Batches", y=1.02, fontsize=16)
        plt.tight_layout()
        display(fig)
        plt.close()

    # Display CSV files with Rich formatting
    console = Console()
    for file in csv_files:
        file_path = os.path.join(session_dir, file)
        try:
            df = pd.read_csv(file_path)
            table = Table(title=f"\n📊 {file}", show_header=True, header_style="bold magenta")
            
            # Add columns
            for column in df.columns:
                table.add_column(column, justify="right", style="cyan")
            
            # Add rows with alternating colors
            for idx, row in df.iterrows():
                style = "dim" if idx % 2 == 0 else "none"
                table.add_row(*[str(val) for val in row], style=style)
            
            console.print(table)
            print("\n")  # Add spacing between tables
            
        except Exception as e:
            logging.error(f"Error reading CSV file {file}: {e}")

    # # Display text and YAML files
    # for file in text_files + yaml_files:
    #     file_path = os.path.join(session_dir, file)
    #     try:
    #         with open(file_path, 'r') as f:
    #             content = f.read()
    #             if file.lower().endswith('.yaml'):
    #                 content = yaml.safe_load(f)
    #             console.print(f"\n📄 {file}", style="bold blue")
    #             console.print("─" * 80)
    #             if isinstance(content, (dict, list)):
    #                 console.print(content, style="yellow")
    #             else:
    #                 console.print(content)
    #             console.print("─" * 80 + "\n")
    #     except Exception as e:
    #         logging.error(f"Error reading file {file}: {e}")