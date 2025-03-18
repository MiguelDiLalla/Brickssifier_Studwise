"""
LEGO Bricks ML Vision - Multiclass Training Pipeline
Specialized script for training multiclass LEGO brick detection model.

Key features:
- Fixed dataset path from presentation/Datasets_Compress/multiclass_dataset.zip 
- Dynamic class mapping extraction from metadata JSON
- Data augmentation and splitting
- Progress tracking with rich logging
"""

import os
import json
import shutil
import logging
import zipfile
import random
from pathlib import Path
from datetime import datetime
import yaml
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from rich.logging import RichHandler
from rich.progress import Progress
from rich.console import Console
from rich.prompt import Confirm
import click

# Configure rich console and click
console = Console()
click.rich_click.USE_RICH = True

def setup_logging():
    """Configure rich logging with emojis"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def detect_hardware():
    """Detect available hardware for training"""
    if torch.cuda.is_available():
        device_list = list(range(torch.cuda.device_count()))
        devices = ','.join(map(str, device_list))
        logging.info(f"🖥️ Using GPU(s): {devices}")
        return devices
    else:
        if Confirm.ask("⚠️ No GPU detected. Training on CPU can be very slow. Continue?"):
            logging.info("🖥️ Using CPU for training")
            return "cpu"
        else:
            raise SystemExit("Training cancelled by user")

def extract_dataset(repo_root: Path) -> Path:
    """Extract multiclass dataset from zip"""
    source = repo_root / "presentation" / "Datasets_Compress" / "multiclass_dataset.zip"
    destination = repo_root / "cache" / "datasets" / "multiclass"
    
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found at {source}")
        
    # Clean destination if exists
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    
    with Progress() as progress:
        task = progress.add_task("📦 Extracting dataset...", total=None)
        with zipfile.ZipFile(source, "r") as zip_ref:
            zip_ref.extractall(destination)
        progress.update(task, completed=True)
    
    logging.info(f"✅ Dataset extracted to {destination}")
    return destination

def create_dataset_structure(dataset_path: Path, train_ratio=0.7, val_ratio=0.2):
    """Create and organize YOLO dataset structure with splits"""
    splits = ["train", "val", "test"]
    
    # Create directories
    for split in splits:
        (dataset_path / "dataset" / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / "dataset" / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get files and check pairs
    images = list((dataset_path / "images").glob("*.jpg"))
    
    valid_pairs = []
    for img_path in images:
        label_path = dataset_path / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    if not valid_pairs:
        raise ValueError("No valid image-label pairs found")
    
    # Random split
    random.shuffle(valid_pairs)
    n_train = int(len(valid_pairs) * train_ratio)
    n_val = int(len(valid_pairs) * val_ratio)
    
    splits_dict = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:]
    }
    
    # Move files to splits
    with Progress() as progress:
        task = progress.add_task("📊 Creating dataset splits...", total=len(valid_pairs))
        
        for split, pairs in splits_dict.items():
            for img_path, label_path in pairs:
                # Move image
                dst_img = dataset_path / "dataset" / "images" / split / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Move label
                dst_label = dataset_path / "dataset" / "labels" / split / label_path.name
                shutil.copy2(label_path, dst_label)
                
                progress.advance(task)
    
    logging.info(f"✅ Dataset split: {len(splits_dict['train'])} train, "
                f"{len(splits_dict['val'])} val, {len(splits_dict['test'])} test")
    return dataset_path / "dataset"

def get_classes(dataset_path: Path) -> list:
    """Extract class names from metadata JSON"""
    metadata_path = dataset_path / "labels" / "batch_inference_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError("Dataset metadata not found")
        
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    classes = metadata["config"]["classes"]
    logging.info(f"📋 Found {len(classes)} classes: {', '.join(classes)}")
    return classes

def create_dataset_yaml(dataset_path: Path, classes: list) -> Path:
    """Create YOLO dataset configuration file"""
    config = {
        "path": str(dataset_path),
        "train": "images/train", 
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes
    }
    
    yaml_path = dataset_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logging.info(f"✅ Created dataset config at {yaml_path}")
    return yaml_path

def train_model(yaml_path: Path, device: str, epochs: int = 100, batch_size: int = 16):
    """Train YOLOv8 model on multiclass dataset"""
    model = YOLO("yolov8n.pt")
    
    # Setup training output directory
    results_dir = Path.cwd() / "results" / "multiclass"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    training_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    with Progress() as progress:
        task = progress.add_task("🚀 Training model...", total=epochs)
        
        try:
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                batch=batch_size,
                device=device,
                project=str(results_dir),
                name=training_name,
                exist_ok=True,
                pretrained=True,
                patience=5,
                verbose=True
            )
            progress.update(task, completed=epochs)
            logging.info("✅ Training completed successfully")
            return results_dir / training_name
            
        except Exception as e:
            logging.error(f"❌ Training failed: {str(e)}")
            raise

@click.group()
def cli():
    """🚀 LEGO Bricks ML Vision - Multiclass Training Pipeline

    This tool provides commands for training a YOLOv8 model on multiclass LEGO brick detection.
    The pipeline includes dataset preparation, training, and cleanup functionality.

    Example usage:
        # Train a model with default parameters
        python train_multiclass.py train

        # Train with custom parameters
        python train_multiclass.py train --epochs 200 --batch-size 32

        # Clean up training artifacts
        python train_multiclass.py cleanup
    """
    pass

@cli.command()
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=16, help='Training batch size')
@click.option('--train-ratio', default=0.7, type=float, help='Ratio of data for training (0-1)')
@click.option('--val-ratio', default=0.2, type=float, help='Ratio of data for validation (0-1)')
@click.option('--force-gpu', is_flag=True, help='Force GPU usage, exit if not available')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def train(epochs, batch_size, train_ratio, val_ratio, force_gpu, yes):
    """🎯 Train a YOLOv8 model for multiclass LEGO brick detection.

    This command executes the complete training pipeline:
    1. Extracts the multiclass dataset from presentation/Datasets_Compress
    2. Creates YOLO-compatible dataset structure with train/val/test splits
    3. Configures and trains a YOLOv8 model
    4. Saves results and model weights

    The training process includes:
    - Automatic hardware detection (GPU/CPU)
    - Progress tracking with rich formatting
    - Early stopping for optimal results
    - Comprehensive logging
    
    Example usage:
        train --epochs 150 --batch-size 32 --train-ratio 0.8 --val-ratio 0.1
    """
    setup_logging()
    repo_root = Path(__file__).parent
    
    try:
        # Hardware setup with force-gpu option
        if force_gpu and not torch.cuda.is_available():
            raise SystemExit("GPU required but not available")
        
        if not yes and not force_gpu:
            device = detect_hardware()
        else:
            device = "0" if torch.cuda.is_available() else "cpu"
        
        # Validate split ratios
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("Train and validation ratios must sum to less than 1.0")
        
        # Dataset preparation
        dataset_path = extract_dataset(repo_root)
        classes = get_classes(dataset_path)
        
        # Create YOLO structure
        yolo_dataset = create_dataset_structure(dataset_path, train_ratio, val_ratio)
        yaml_path = create_dataset_yaml(yolo_dataset, classes)
        
        # Train model
        results_dir = train_model(yaml_path, device, epochs, batch_size)
        logging.info(f"✨ Training pipeline completed. Results saved to {results_dir}")
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        raise SystemExit(1)

@cli.command()
@click.option('--all', '-a', 'clean_all', is_flag=True, help='Remove all cache and results')
@click.option('--cache', is_flag=True, help='Remove only cache directory')
@click.option('--results', is_flag=True, help='Remove only results directory')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def cleanup(clean_all, cache, results, yes):
    """🧹 Clean up training artifacts and temporary files.

    This command helps manage disk space by removing training artifacts:
    - Cache directory: Contains extracted datasets and temporary files
    - Results directory: Contains training results, models, and logs

    You can specify which directories to clean:
    --all     : Remove both cache and results directories
    --cache   : Remove only the cache directory
    --results : Remove only the results directory

    Example usage:
        cleanup --all -y  # Remove everything without confirmation
        cleanup --cache   # Remove only cache with confirmation
    """
    setup_logging()
    repo_root = Path(__file__).parent
    
    # Determine directories to clean
    to_clean = []
    if clean_all or cache:
        to_clean.append(repo_root / "cache")
    if clean_all or results:
        to_clean.append(repo_root / "results")
    
    if not to_clean:
        logging.error("❌ Please specify what to clean: --all, --cache, or --results")
        return
    
    # Confirm cleanup
    if not yes:
        dirs_str = ", ".join(str(d) for d in to_clean)
        if not Confirm.ask(f"⚠️ This will remove: {dirs_str}. Continue?"):
            logging.info("Cleanup cancelled")
            return
    
    # Perform cleanup
    for path in to_clean:
        if path.exists():
            shutil.rmtree(path)
            logging.info(f"✅ Removed {path}")
        else:
            logging.warning(f"⚠️ Directory not found: {path}")
    
    logging.info("🧹 Cleanup completed")

if __name__ == "__main__":
    cli()