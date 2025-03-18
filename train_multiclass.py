"""
LEGO Bricks ML Vision - Multiclass Training Pipeline
Specialized script for training multiclass LEGO brick detection model.

Key features:
- Fixed dataset path from presentation/Datasets_Compress/multiclass_dataset.zip 
- Dynamic class mapping extraction from metadata JSON
- Configurable data augmentation
- Multi-GPU support with CPU fallback confirmation
- Rich progress tracking and logging
"""

import os
import json
import shutil
import logging
import zipfile
from pathlib import Path
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn
from rich.console import Console
from rich.prompt import Confirm
import torch
import yaml
from ultralytics import YOLO
import albumentations as A
import cv2
import click
import rich_click
from datetime import datetime

# Configure rich_click for prettier CLI
rich_click.USE_RICH = True
console = Console()

# =============================================================================
# Logging Setup with Rich
# =============================================================================

def setup_logging():
    """Configure rich logging with emojis and color coding"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
# =============================================================================
# Dataset Preparation
# =============================================================================

def detect_hardware():
    """Detect available hardware and handle CPU confirmation"""
    if torch.cuda.is_available():
        device_list = list(range(torch.cuda.device_count()))
        devices = ','.join(map(str, device_list))
        logging.info(f"🖥️ Using GPU(s): {devices}")
        return devices
    else:
        if Confirm.ask("⚠️ No GPU detected. Training on CPU can be very slow. Do you want to continue?"):
            logging.info("🖥️ Using CPU for training")
            return "cpu"
        else:
            raise click.Abort()

def get_classes_from_metadata(labels_dir):
    """Extract class mapping from dataset metadata JSON"""
    metadata_path = os.path.join(labels_dir, "batch_inference_metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    classes = metadata['config']['classes']
    logging.info(f"📋 Found {len(classes)} classes in metadata")
    return classes

def unzip_multiclass_dataset(repo_root, force=False):
    """Extract multiclass dataset with progress tracking"""
    source = os.path.join(repo_root, "presentation", "Datasets_Compress", "multiclass_dataset.zip")
    destination = os.path.join(repo_root, "cache", "datasets", "multiclass")
    
    if os.path.exists(destination) and not force:
        logging.info("📂 Using cached dataset")
        return destination
        
    os.makedirs(destination, exist_ok=True)
    
    with Progress() as progress:
        task = progress.add_task("📦 Extracting dataset...", total=None)
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(destination)
        progress.update(task, completed=True)
    
    logging.info("✅ Dataset extracted successfully")
    return destination

def create_dataset_yaml(dataset_path, classes):
    """Create YOLO dataset configuration YAML"""
    yaml_config = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes
    }
    
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    logging.info("✅ Dataset YAML configuration created")
    return yaml_path

def configure_augmentation(mode="medium"):
    """Configure augmentation pipeline based on intensity"""
    augmentation_configs = {
        "light": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]),
        "medium": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.GaussianBlur(p=0.2),
        ]),
        "heavy": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.Rotate(limit=30, p=0.7),
            A.GaussianBlur(p=0.3),
            A.RandomGamma(p=0.2),
            A.CLAHE(p=0.3),
        ])
    }
    return augmentation_configs[mode]

def train_model(dataset_yaml, device, epochs, batch_size, augment_mode):
    """Train YOLOv8 model for multiclass detection"""
    model = YOLO('yolov8n.pt')
    
    with Progress() as progress:
        task = progress.add_task("🚀 Training model...", total=epochs)
        
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            device=device,
            project="results/multiclass",
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M')}",
            exist_ok=True,
            pretrained=True,
            patience=7,
            augment=True,
            verbose=True,
            classes=None  # Use all classes from dataset
        )
        
        # Update progress after training
        progress.update(task, completed=epochs)
    
    logging.info("✅ Training completed successfully")
    return results

def zip_results(results_path, repo_root):
    """Zip training results for easy sharing"""
    output_path = os.path.join(repo_root, "..", "multiclass_training_results.zip")
    
    with Progress() as progress:
        task = progress.add_task("📦 Compressing results...", total=None)
        shutil.make_archive(output_path[:-4], 'zip', results_path)
        progress.update(task, completed=True)
    
    logging.info(f"✅ Results compressed to: {output_path}")
    return output_path

def cleanup(repo_root):
    """Clean up temporary files and folders"""
    paths = [
        os.path.join(repo_root, "cache"),
        os.path.join(repo_root, "results")
    ]
    
    with Progress() as progress:
        task = progress.add_task("🧹 Cleaning up...", total=len(paths))
        
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            progress.advance(task)
    
    logging.info("✅ Cleanup completed")

# =============================================================================
# CLI Interface
# =============================================================================

@click.command()
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=16, help='Training batch size')
@click.option('--augment-mode', type=click.Choice(['light', 'medium', 'heavy']), 
              default='medium', help='Augmentation intensity')
@click.option('--force-extract', is_flag=True, help='Force dataset re-extraction')
@click.option('--cleanup', is_flag=True, help='Clean up after training')
def main(epochs, batch_size, augment_mode, force_extract, cleanup):
    """Train multiclass LEGO brick detection model"""
    setup_logging()
    repo_root = Path(__file__).parent
    
    try:
        # Hardware detection with CPU confirmation
        device = detect_hardware()
        
        # Dataset preparation
        dataset_path = unzip_multiclass_dataset(repo_root, force=force_extract)
        classes = get_classes_from_metadata(os.path.join(dataset_path, "labels"))
        yaml_path = create_dataset_yaml(dataset_path, classes)
        
        # Training
        results = train_model(
            dataset_yaml=yaml_path,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            augment_mode=augment_mode
        )
        
        # Results compression
        zip_results(os.path.join(repo_root, "results", "multiclass"), repo_root)
        
        if cleanup:
            cleanup(repo_root)
            
        logging.info("✨ Training pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise click.Abort()

if __name__ == '__main__':
    main()