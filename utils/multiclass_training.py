import logging
import os
import shutil
from pathlib import Path
from typing import Tuple, List

import albumentations as A
import cv2
import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from ultralytics import YOLO

console = Console()

class MulticlassTrainer:
    """Handles training pipeline for multiclass YOLO dataset."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path.cwd() / "cache" / "multiclass"
        self.console = Console()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("multiclass_trainer")
        self.logger.setLevel(logging.INFO)
        
        # Add handlers if they don't exist
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(log_dir / "multiclass_training.log")
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

    def validate_dataset(self) -> Tuple[Path, Path]:
        """Validate image-label parity and return paths."""
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Dataset structure invalid at {self.dataset_path}")
        
        image_files = set(f.stem for f in images_dir.glob("*.jpg"))
        label_files = set(f.stem for f in labels_dir.glob("*.txt"))
        
        if not image_files:
            raise ValueError("No images found in dataset")
            
        if image_files != label_files:
            missing = image_files - label_files
            extra = label_files - image_files
            raise ValueError(f"Mismatched files: {len(missing)} missing labels, {len(extra)} extra labels")
            
        self.logger.info(f"Dataset validated: {len(image_files)} image-label pairs")
        return images_dir, labels_dir

    def create_yolo_structure(self) -> Path:
        """Create YOLO dataset structure in cache directory."""
        dataset_dir = self.cache_dir / "dataset"
        splits = ['train', 'val', 'test']
        
        for split in splits:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Created YOLO structure at {dataset_dir}")
        return dataset_dir

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2):
        """Split dataset into train/val/test sets."""
        images_dir, labels_dir = self.validate_dataset()
        dataset_dir = self.create_yolo_structure()
        
        image_files = list(images_dir.glob("*.jpg"))
        np.random.shuffle(image_files)
        
        n = len(image_files)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': image_files[:train_idx],
            'val': image_files[train_idx:val_idx],
            'test': image_files[val_idx:]
        }
        
        with Progress() as progress:
            task = progress.add_task("Splitting dataset...", total=n)
            
            for split_name, files in splits.items():
                for img_path in files:
                    # Copy image
                    shutil.copy2(
                        img_path, 
                        dataset_dir / 'images' / split_name / img_path.name
                    )
                    # Copy corresponding label
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    shutil.copy2(
                        label_path,
                        dataset_dir / 'labels' / split_name / f"{img_path.stem}.txt"
                    )
                    progress.advance(task)
                    
        self.logger.info(f"Dataset split: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return dataset_dir

    def augment_training_data(self, multiplier: int = 2):
        """Augment training data using Albumentations."""
        train_imgs_dir = self.cache_dir / "dataset" / "images" / "train"
        train_labels_dir = self.cache_dir / "dataset" / "labels" / "train"
        
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        image_files = list(train_imgs_dir.glob("*.jpg"))
        
        with Progress() as progress:
            task = progress.add_task("Augmenting training data...", 
                                   total=len(image_files) * multiplier)
            
            for img_path in image_files:
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Read YOLO labels
                label_path = train_labels_dir / f"{img_path.stem}.txt"
                bboxes = []
                class_labels = []
                
                with open(label_path) as f:
                    for line in f:
                        class_id, x, y, w, h = map(float, line.strip().split())
                        bboxes.append([x, y, w, h])
                        class_labels.append(class_id)
                
                # Generate augmentations
                for i in range(multiplier):
                    transformed = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    # Save augmented image
                    aug_img_path = train_imgs_dir / f"{img_path.stem}_aug{i}.jpg"
                    cv2.imwrite(
                        str(aug_img_path),
                        cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                    )
                    
                    # Save augmented labels
                    aug_label_path = train_labels_dir / f"{img_path.stem}_aug{i}.txt"
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(transformed['bboxes'], 
                                                transformed['class_labels']):
                            f.write(f"{int(class_id)} {' '.join(map(str, bbox))}\n")
                            
                    progress.advance(task)
        
        self.logger.info(f"Augmentation complete: {len(image_files) * multiplier} "
                        f"new images created")

    def train(self, epochs: int = 50, patience: int = 3):
        """Train YOLO model with early stopping."""
        # Create YAML config
        data_yaml = self.cache_dir / "dataset.yaml"
        config = {
            'path': str(self.cache_dir),
            'train': 'dataset/images/train',
            'val': 'dataset/images/val',
            'test': 'dataset/images/test',
            'names': self._get_class_names()
        }
        
        with open(data_yaml, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize and train model
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            patience=patience,
            batch=16,
            imgsz=640,
            device='0',  # Use first GPU
            project=str(self.cache_dir),
            name='train'
        )
        
        self._display_results(results)
        return results

    def _get_class_names(self) -> List[str]:
        """Extract class names from labels."""
        label_file = next((self.dataset_path / "labels").glob("*.txt"))
        classes = set()
        
        with open(label_file) as f:
            for line in f:
                class_id = int(line.split()[0])
                classes.add(class_id)
                
        return [f"class_{i}" for i in range(max(classes) + 1)]

    def _display_results(self, results):
        """Display training results using Rich."""
        console = Console()
        
        metrics = results.results_dict
        console.print(Panel(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"Best mAP@50: {metrics['metrics/mAP50(B)']:.3f}\n"
            f"Best mAP@50-95: {metrics['metrics/mAP50-95(B)']:.3f}\n"
            f"Final Precision: {metrics.get('metrics/precision', 0):.3f}\n"
            f"Final Recall: {metrics.get('metrics/recall', 0):.3f}\n",
            title="Training Results",
            border_style="green"
        ))

def train_multiclass_model(dataset_path: str):
    """Main entry point for multiclass training."""
    trainer = MulticlassTrainer(dataset_path)
    
    try:
        console.print("[bold]Starting multiclass training pipeline[/bold]")
        
        with Progress() as progress:
            task = progress.add_task("Validating dataset...", total=4)
            trainer.validate_dataset()
            progress.advance(task)
            
            progress.update(task, description="Splitting dataset...")
            trainer.split_dataset()
            progress.advance(task)
            
            progress.update(task, description="Augmenting training data...")
            trainer.augment_training_data()
            progress.advance(task)
            
            progress.update(task, description="Training model...")
            results = trainer.train()
            progress.advance(task)
            
        return results
        
    except Exception as e:
        console.print(f"[bold red]Error during training: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    train_multiclass_model("path/to/dataset")
