"""
LEGO Bricks ML Vision - Single-class Training Pipeline
Specialized script for training single-class LEGO detection models (bricks or studs).

Key features:
- Dual mode training for bricks or studs detection
- Fixed dataset paths from presentation/Datasets_Compress/{mode}_dataset.zip 
- Single class configuration for each mode
- Data augmentation and splitting
- Progress tracking with rich logging
"""

import os
import shutil
import logging
import zipfile
import random
from pathlib import Path
from datetime import datetime
import yaml
import torch
from ultralytics import YOLO
from rich.logging import RichHandler
from rich.progress import Progress
from rich.console import Console
from rich.prompt import Confirm
import click
import rich_click

# Configure rich console and rich-click
console = Console()
rich_click.USE_RICH = True

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

def extract_dataset(repo_root: Path, mode: str) -> Path:
    """Extract single-class dataset from zip based on mode"""
    source = repo_root / "presentation" / "Datasets_Compress" / f"{mode}_dataset.zip"
    destination = repo_root / "cache" / "datasets" / mode
    
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
    
    # Find subfolders
    subfolders = [f for f in os.listdir(dataset_path) 
                  if os.path.isdir(dataset_path / f)]
    
    # Detect images and labels folders
    images_path = None
    labels_path = None
    
    for folder in subfolders:
        folder_path = dataset_path / folder
        sample_files = os.listdir(folder_path)[:5]  # Check first 5 files
        
        # Check if folder contains images
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in sample_files):
            images_path = folder_path
        elif any(f.lower().endswith('.txt') for f in sample_files):
            labels_path = folder_path
    
    if not images_path or not labels_path:
        raise ValueError("Could not identify images and labels folders in dataset")
        
    logging.info(f"Found image folder: {images_path}")
    logging.info(f"Found labels folder: {labels_path}")
    
    # Get files and check pairs with proper path handling
    images = list(Path(images_path).glob("*.jpg"))
    
    valid_pairs = []
    for img_path in images:
        label_path = Path(labels_path) / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    if not valid_pairs:
        raise ValueError(f"No valid image-label pairs found in {images_path} and {labels_path}")
    
    # Random split
    random.shuffle(valid_pairs)
    n_train = int(len(valid_pairs) * train_ratio)
    n_val = int(len(valid_pairs) * val_ratio)
    
    splits_dict = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:]
    }
    
    # Move files to splits with proper path handling
    with Progress() as progress:
        task = progress.add_task("📊 Creating dataset splits...", total=len(valid_pairs))
        
        for split, pairs in splits_dict.items():
            for img_path, label_path in pairs:
                # Move image
                dst_img = dataset_path / "dataset" / "images" / split / img_path.name
                shutil.copy2(str(img_path), str(dst_img))
                
                # Move label
                dst_label = dataset_path / "dataset" / "labels" / split / label_path.name
                shutil.copy2(str(label_path), str(dst_label))
                
                progress.advance(task)
    
    logging.info(f"✅ Dataset split: {len(splits_dict['train'])} train, "
                f"{len(splits_dict['val'])} val, {len(splits_dict['test'])} test")
    return dataset_path / "dataset"

def create_dataset_yaml(dataset_path: Path, mode: str) -> Path:
    """Create YOLO dataset configuration file"""
    config = {
        "path": str(dataset_path),
        "train": "images/train", 
        "val": "images/val",
        "test": "images/test",
        "nc": 1,  # Single class
        "names": [mode]  # Class name is the mode (bricks or studs)
    }
    
    yaml_path = dataset_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml_content = yaml.dump(config, default_flow_style=False)
        f.write(yaml_content)
    
    logging.info(f"✅ Created dataset config at {yaml_path}")
    logging.info("📄 Dataset YAML content:\n" + yaml_content)
    return yaml_path

def train_model(yaml_path: Path, mode: str, device: str, epochs: int = 100, batch_size: int = 16):
    """Train YOLOv8 model on single-class dataset"""
    model = YOLO("yolov8n.pt")
    
    # Setup training output directory
    results_dir = Path.cwd() / "results" / mode
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
            logging.info("✅ Training completed successfully")
            return results_dir / training_name
            
        except Exception as e:
            logging.error(f"❌ Training failed: {str(e)}")
            raise

def zip_results(results_dir: Path, mode: str) -> Path:
    """Zip training results and save outside repository"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    zip_name = f"{mode}_{timestamp}_results.zip"
    
    # Get parent directory of repository for saving zip
    repo_root = Path(__file__).parent
    output_dir = repo_root.parent
    zip_path = output_dir / zip_name
    
    with Progress() as progress:
        task = progress.add_task("📦 Zipping results...", total=None)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(results_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(results_dir)
                    zipf.write(file_path, arcname)
                    
        progress.update(task, completed=True)
    
    logging.info(f"📦 Results archived to: {zip_path}")
    return zip_path

def check_and_clean_directories(repo_root: Path, yes: bool = False) -> None:
    """Check if cache or results directories have content and clean if needed"""
    dirs_to_check = {
        "cache": repo_root / "cache",
        "results": repo_root / "results"
    }
    
    non_empty_dirs = []
    for name, dir_path in dirs_to_check.items():
        if dir_path.exists() and any(dir_path.iterdir()):
            non_empty_dirs.append(name)
    
    if non_empty_dirs:
        dirs_str = " and ".join(non_empty_dirs)
        if yes or Confirm.ask(f"⚠️ {dirs_str} directories are not empty. Clean them before proceeding?"):
            for dir_name in non_empty_dirs:
                shutil.rmtree(dirs_to_check[dir_name])
                dirs_to_check[dir_name].mkdir(exist_ok=True)
                logging.info(f"✅ Cleaned {dir_name} directory")
        else:
            raise SystemExit("Training cancelled: directories must be empty to proceed")
    else:
        for dir_path in dirs_to_check.values():
            dir_path.mkdir(exist_ok=True)
        logging.info("✅ Working directories are clean")

@click.group()
def cli():
    """🚀 LEGO Bricks ML Vision - Single-class Training Pipeline

    This tool provides commands for training YOLOv8 models for single-class detection
    of either LEGO bricks or studs. The pipeline includes dataset preparation,
    training, and cleanup functionality.

    Example usage:
        # Train a brick detection model with default parameters
        python train_singleclass.py train --mode bricks

        # Train a stud detection model with custom parameters
        python train_singleclass.py train --mode studs --epochs 200 --batch-size 32

        # Clean up training artifacts
        python train_singleclass.py cleanup
    """
    pass

@cli.command()
@click.option('--mode', type=click.Choice(['bricks', 'studs'], case_sensitive=False), 
              required=True, help='Training mode: bricks or studs detection')
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=16, help='Training batch size')
@click.option('--train-ratio', default=0.7, type=float, help='Ratio of data for training (0-1)')
@click.option('--val-ratio', default=0.2, type=float, help='Ratio of data for validation (0-1)')
@click.option('--force-gpu', is_flag=True, help='Force GPU usage, exit if not available')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def train(mode, epochs, batch_size, train_ratio, val_ratio, force_gpu, yes):
    """🎯 Train a YOLOv8 model for single-class LEGO detection.

    This command executes the complete training pipeline:
    1. Extracts the dataset from presentation/Datasets_Compress/{mode}_dataset.zip
    2. Creates YOLO-compatible dataset structure with train/val/test splits
    3. Configures and trains a YOLOv8 model
    4. Saves results and model weights

    The training process includes:
    - Automatic hardware detection (GPU/CPU)
    - Progress tracking with rich formatting
    - Early stopping for optimal results
    - Comprehensive logging
    
    Example usage:
        train --mode bricks --epochs 150 --batch-size 32 --train-ratio 0.8
        train --mode studs --epochs 100 --batch-size 16 --force-gpu
    """
    setup_logging()
    repo_root = Path(__file__).parent
    
    try:
        # Check and clean directories if needed
        check_and_clean_directories(repo_root, yes)
        
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
        dataset_path = extract_dataset(repo_root, mode)
        
        # Create YOLO structure
        yolo_dataset = create_dataset_structure(dataset_path, train_ratio, val_ratio)
        yaml_path = create_dataset_yaml(yolo_dataset, mode)
        
        # Train model
        results_dir = train_model(yaml_path, mode, device, epochs, batch_size)
        logging.info(f"✨ Training pipeline completed. Results saved to {results_dir}")
        
        # Archive results
        zip_path = zip_results(results_dir, mode)
        logging.info(f"✅ Training pipeline completed and results archived to {zip_path}")
        
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