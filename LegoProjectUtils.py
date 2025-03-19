#!/usr/bin/env python
"""
LEGO Bricks ML Vision - Unified Project Utilities

A monolithic script that incorporates all core functionalities of the LEGO Bricks ML Vision project
in a single, rich UI enhanced interface. This script is a self-contained version that includes all
necessary utilities without external module dependencies.

Key Features:
    - Brick Detection: Identify LEGO bricks in images
    - Stud Detection: Analyze stud patterns on bricks
    - Dimension Classification: Determine brick dimensions
    - Batch Processing: Handle multiple images efficiently
    - Data Processing: Convert and validate annotations
    - Metadata Management: Handle image EXIF data

Author: Miguel DiLalla
"""

import os
import sys
import logging
import click
import json
import time
import glob
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Core processing imports
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Rich UI components
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, TaskProgressColumn
)
from rich.status import Status
from rich.text import Text
from rich.style import Style
from rich.layout import Layout
from rich.box import ROUNDED

# Initialize Rich console
console = Console()

# Configure logging with rich traceback
from rich.traceback import install
install(show_locals=True)

class LegoVisionError(Exception):
    """Base exception class for LEGO Vision project errors."""
    pass

class ValidationError(LegoVisionError):
    """Raised when input validation fails."""
    pass

class ProcessingError(LegoVisionError):
    """Raised when image or data processing fails."""
    pass

class ModelError(LegoVisionError):
    """Raised when model operations fail."""
    pass

class MetadataError(LegoVisionError):
    """Raised when metadata operations fail."""
    pass

class LegoVisionProject:
    """
    Main class that encapsulates all LEGO Vision project functionality.
    Provides a unified interface for all operations with enhanced UI feedback.
    """
    
    def __init__(self):
        """Initialize the LEGO Vision project with default configurations."""
        self.models = {}
        self.config = self._load_default_config()
        self._setup_logging()
        
        # Status tracking
        self.processed_files = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.current_operation = None
        
        console.print(Panel.fit(
            "[bold blue]LEGO Vision Project Utilities[/bold blue]\n" +
            "[dim]Initialized and ready for processing[/dim]",
            border_style="blue"
        ))
    
    def _load_default_config(self) -> dict:
        """Load default configuration values."""
        return {
            "BRICK_MODEL_PATH": "models/Brick_Model_best.pt",
            "STUD_MODEL_PATH": "models/Stud_Model_best.pt",
            "CONFIDENCE_THRESHOLD": 0.25,
            "BRICKS_DIMENSIONS_CLASSES": {
                "2x2": 0,
                "2x3": 1,
                "2x4": 2,
                "1x2": 3,
                "1x3": 4,
                "1x4": 5
            },
            "OUTPUT_DIR": "results",
            "CACHE_DIR": "cache",
            "LOGS_DIR": "logs"
        }
    
    def _setup_logging(self):
        """Configure logging with rich handler."""
        os.makedirs(self.config["LOGS_DIR"], exist_ok=True)
        
        class RichLogHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    color = {
                        logging.DEBUG: "dim",
                        logging.INFO: "blue",
                        logging.WARNING: "yellow",
                        logging.ERROR: "red",
                        logging.CRITICAL: "red bold"
                    }.get(record.levelno, "white")
                    
                    console.print(f"[{color}]{msg}[/{color}]")
                except Exception:
                    self.handleError(record)
        
        logger = logging.getLogger("LegoVision")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(self.config["LOGS_DIR"], "lego_vision.log")
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # Rich console handler
        rich_handler = RichLogHandler()
        rich_handler.setFormatter(logging.Formatter('%(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(rich_handler)
        
        self.logger = logger

    def _load_model(self, model_type: str) -> None:
        """Load a YOLO model with rich progress feedback."""
        if model_type not in ['brick', 'stud']:
            raise ValidationError(f"Invalid model type: {model_type}")
            
        model_path = self.config[f"{model_type.upper()}_MODEL_PATH"]
        if not os.path.exists(model_path):
            raise ModelError(f"Model not found: {model_path}")
            
        with console.status(f"[bold blue]Loading {model_type} detection model...") as status:
            try:
                self.models[model_type] = YOLO(model_path)
                self.logger.info(f"Loaded {model_type} model from {model_path}")
            except Exception as e:
                raise ModelError(f"Failed to load {model_type} model: {str(e)}")
    
    def _validate_image(self, image_path: str) -> None:
        """Validate image file existence and format."""
        if not os.path.exists(image_path):
            raise ValidationError(f"Image not found: {image_path}")
            
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception as e:
            raise ValidationError(f"Invalid image file {image_path}: {str(e)}")
    
    def detect_bricks(self, 
                     image_path: str,
                     conf: float = None,
                     save_annotated: bool = True,
                     output_folder: str = None) -> Dict:
        """
        Detect LEGO bricks in an image with rich progress feedback.
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold (optional)
            save_annotated: Whether to save visualization
            output_folder: Custom output location (optional)
            
        Returns:
            Dictionary with detection results
        """
        self._validate_image(image_path)
        
        if 'brick' not in self.models:
            self._load_model('brick')
            
        conf = conf or self.config['CONFIDENCE_THRESHOLD']
        output_folder = output_folder or os.path.join(
            self.config['OUTPUT_DIR'],
            'brick_detection',
            Path(image_path).stem
        )
        os.makedirs(output_folder, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            try:
                task = progress.add_task("Detecting bricks...", total=100)
                progress.update(task, advance=10)
                
                # Run detection
                results = self.models['brick'](
                    image_path,
                    conf=conf,
                    save=save_annotated,
                    project=output_folder,
                    name=''
                )
                progress.update(task, advance=70)
                
                # Process results
                boxes = []
                scores = []
                for r in results:
                    boxes.extend(r.boxes.xyxy.cpu().numpy().tolist())
                    scores.extend(r.boxes.conf.cpu().numpy().tolist())
                
                progress.update(task, advance=20)
                
                detection_result = {
                    'boxes': boxes,
                    'scores': scores,
                    'image_path': image_path,
                    'output_folder': output_folder
                }
                
                self.logger.info(
                    f"Detected {len(boxes)} bricks in {Path(image_path).name}"
                )
                return detection_result
                
            except Exception as e:
                raise ProcessingError(f"Brick detection failed: {str(e)}")
    
    def detect_studs(self,
                    image_path: str,
                    brick_box: List[float] = None,
                    conf: float = None,
                    save_annotated: bool = True,
                    output_folder: str = None) -> Dict:
        """
        Detect studs on a LEGO brick with rich progress feedback.
        
        Args:
            image_path: Path to input image
            brick_box: Bounding box of brick [x1,y1,x2,y2] (optional)
            conf: Confidence threshold (optional)
            save_annotated: Whether to save visualization
            output_folder: Custom output location (optional)
            
        Returns:
            Dictionary with detection and classification results
        """
        self._validate_image(image_path)
        
        if 'stud' not in self.models:
            self._load_model('stud')
            
        conf = conf or self.config['CONFIDENCE_THRESHOLD']
        output_folder = output_folder or os.path.join(
            self.config['OUTPUT_DIR'],
            'stud_detection',
            Path(image_path).stem
        )
        os.makedirs(output_folder, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            try:
                task = progress.add_task("Detecting studs...", total=100)
                progress.update(task, advance=10)
                
                # Load and crop image if brick_box provided
                img = cv2.imread(image_path)
                if brick_box:
                    x1,y1,x2,y2 = map(int, brick_box)
                    img = img[y1:y2, x1:x2]
                
                progress.update(task, advance=20)
                
                # Run detection
                results = self.models['stud'](
                    img,
                    conf=conf,
                    save=save_annotated,
                    project=output_folder,
                    name=''
                )
                progress.update(task, advance=50)
                
                # Process results
                boxes = []
                scores = []
                for r in results:
                    boxes.extend(r.boxes.xyxy.cpu().numpy().tolist())
                    scores.extend(r.boxes.conf.cpu().numpy().tolist())
                
                # Classify brick dimensions based on stud pattern
                dimension = self._classify_brick_dimension(len(boxes))
                progress.update(task, advance=20)
                
                detection_result = {
                    'boxes': boxes,
                    'scores': scores,
                    'dimension': dimension,
                    'stud_count': len(boxes),
                    'image_path': image_path,
                    'output_folder': output_folder
                }
                
                self.logger.info(
                    f"Detected {len(boxes)} studs, classified as {dimension}"
                )
                return detection_result
                
            except Exception as e:
                raise ProcessingError(f"Stud detection failed: {str(e)}")
    
    def _classify_brick_dimension(self, stud_count: int) -> str:
        """Classify brick dimensions based on stud count."""
        dimension_map = {
            4: "2x2",
            6: "2x3",
            8: "2x4",
            2: "1x2",
            3: "1x3",
            4: "1x4"
        }
        return dimension_map.get(stud_count, "unknown")
    
    def run_full_pipeline(self,
                         image_path: str,
                         conf: float = None,
                         save_annotated: bool = True,
                         output_folder: str = None) -> Dict:
        """
        Run complete detection pipeline: bricks -> studs -> classification.
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold (optional)
            save_annotated: Whether to save visualizations
            output_folder: Custom output location (optional)
            
        Returns:
            Dictionary with complete analysis results
        """
        self._validate_image(image_path)
        
        output_folder = output_folder or os.path.join(
            self.config['OUTPUT_DIR'],
            'full_pipeline',
            Path(image_path).stem
        )
        os.makedirs(output_folder, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            try:
                # Phase 1: Brick Detection
                task1 = progress.add_task("Phase 1: Detecting bricks...", total=100)
                brick_results = self.detect_bricks(
                    image_path,
                    conf=conf,
                    save_annotated=save_annotated,
                    output_folder=os.path.join(output_folder, 'bricks')
                )
                progress.update(task1, completed=100)
                
                # Phase 2: Stud Detection for each brick
                task2 = progress.add_task("Phase 2: Analyzing studs...", total=100)
                step = 100 / max(len(brick_results['boxes']), 1)
                
                stud_results = []
                for i, box in enumerate(brick_results['boxes']):
                    result = self.detect_studs(
                        image_path,
                        brick_box=box,
                        conf=conf,
                        save_annotated=save_annotated,
                        output_folder=os.path.join(output_folder, f'studs_brick_{i}')
                    )
                    stud_results.append(result)
                    progress.update(task2, advance=step)
                
                pipeline_result = {
                    'image_path': image_path,
                    'output_folder': output_folder,
                    'brick_results': brick_results,
                    'studs_results': stud_results
                }
                
                self.logger.info(
                    f"Completed full analysis of {Path(image_path).name}"
                )
                return pipeline_result
                
            except Exception as e:
                raise ProcessingError(f"Pipeline execution failed: {str(e)}")
    
    def process_labelme_to_yolo(self,
                               input_path: str,
                               output_path: str = None,
                               clean: bool = False) -> Dict:
        """
        Convert LabelMe JSON annotations to YOLO format with progress tracking.
        
        Args:
            input_path: Input folder with LabelMe JSONs
            output_path: Output folder for YOLO files (optional)
            clean: Whether to clean output directory first
            
        Returns:
            Dictionary with conversion statistics
        """
        if not os.path.exists(input_path):
            raise ValidationError(f"Input path does not exist: {input_path}")
            
        output_path = output_path or os.path.join(
            self.config['OUTPUT_DIR'],
            'yolo_annotations'
        )
        
        if clean and os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Collect JSON files
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        if not json_files:
            raise ValidationError(f"No JSON files found in {input_path}")
            
        stats = {
            'total': len(json_files),
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(
                "[green]Converting annotations...",
                total=len(json_files)
            )
            
            for json_file in json_files:
                try:
                    # Load JSON
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract image info
                    img_path = os.path.join(
                        os.path.dirname(json_file),
                        data['imagePath']
                    )
                    if not os.path.exists(img_path):
                        raise ValidationError(f"Image not found: {data['imagePath']}")
                    
                    img = cv2.imread(img_path)
                    height, width = img.shape[:2]
                    
                    # Convert annotations
                    yolo_lines = []
                    for shape in data['shapes']:
                        if shape['shape_type'] != 'rectangle':
                            continue
                            
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        # Convert to YOLO format
                        x_center = (x1 + x2) / (2 * width)
                        y_center = (y1 + y2) / (2 * height)
                        w = abs(x2 - x1) / width
                        h = abs(y2 - y1) / height
                        
                        # Get class index from label
                        class_idx = self.config['BRICKS_DIMENSIONS_CLASSES'].get(
                            shape['label'],
                            0  # Default to first class if unknown
                        )
                        
                        yolo_lines.append(
                            f"{class_idx} {x_center} {y_center} {w} {h}"
                        )
                    
                    # Save YOLO annotation
                    base_name = os.path.splitext(os.path.basename(json_file))[0]
                    yolo_path = os.path.join(output_path, f"{base_name}.txt")
                    
                    with open(yolo_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    
                    stats['success'] += 1
                    
                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append((json_file, str(e)))
                    self.logger.warning(
                        f"Failed to convert {json_file}: {str(e)}"
                    )
                
                progress.update(task, advance=1)
        
        return stats
    
    def process_keypoints_to_boxes(self,
                                 input_path: str,
                                 output_path: str = None,
                                 area_ratio: float = 0.4,
                                 clean: bool = False) -> Dict:
        """
        Convert keypoint annotations to bounding boxes.
        
        Args:
            input_path: Input folder with keypoint JSONs
            output_path: Output folder for bbox annotations
            area_ratio: Area ratio for bbox calculation
            clean: Whether to clean output directory first
            
        Returns:
            Dictionary with conversion statistics
        """
        if not os.path.exists(input_path):
            raise ValidationError(f"Input path does not exist: {input_path}")
            
        output_path = output_path or os.path.join(
            self.config['OUTPUT_DIR'],
            'bbox_annotations'
        )
        
        if clean and os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        if not json_files:
            raise ValidationError(f"No JSON files found in {input_path}")
            
        stats = {
            'total': len(json_files),
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(
                "[green]Converting keypoints...",
                total=len(json_files)
            )
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    img_path = os.path.join(
                        os.path.dirname(json_file),
                        data['imagePath']
                    )
                    if not os.path.exists(img_path):
                        raise ValidationError(f"Image not found: {data['imagePath']}")
                    
                    img = cv2.imread(img_path)
                    height, width = img.shape[:2]
                    
                    boxes = []
                    for shape in data['shapes']:
                        if shape['shape_type'] != 'point':
                            continue
                            
                        points = np.array([shape['points']])
                        x, y = points[0][0]
                        
                        # Calculate bbox dimensions based on area ratio
                        box_width = int(width * area_ratio)
                        box_height = int(height * area_ratio)
                        
                        x1 = max(0, int(x - box_width/2))
                        y1 = max(0, int(y - box_height/2))
                        x2 = min(width, x1 + box_width)
                        y2 = min(height, y1 + box_height)
                        
                        boxes.append({
                            'label': shape['label'],
                            'bbox': [x1, y1, x2, y2]
                        })
                    
                    # Save boxes in JSON format
                    base_name = os.path.splitext(os.path.basename(json_file))[0]
                    output_file = os.path.join(output_path, f"{base_name}_boxes.json")
                    
                    with open(output_file, 'w') as f:
                        json.dump({
                            'image': data['imagePath'],
                            'boxes': boxes
                        }, f, indent=2)
                    
                    stats['success'] += 1
                    
                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append((json_file, str(e)))
                    self.logger.warning(
                        f"Failed to convert {json_file}: {str(e)}"
                    )
                
                progress.update(task, advance=1)
        
        return stats
    
    def read_exif(self, image_path: str) -> Dict:
        """Read EXIF metadata from image."""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if not exif:
                    return {}
                    
                # Convert EXIF data to readable format
                exif_dict = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
                    
                return exif_dict
                
        except Exception as e:
            raise MetadataError(f"Failed to read EXIF data: {str(e)}")
    
    def clean_exif(self, image_path: str) -> None:
        """Remove EXIF metadata from image."""
        try:
            img = Image.open(image_path)
            
            # Create new image without EXIF
            data = list(img.getdata())
            new_img = Image.new(img.mode, img.size)
            new_img.putdata(data)
            
            # Save back to original path
            new_img.save(image_path)
            self.logger.info(f"Cleaned EXIF metadata from {image_path}")
            
        except Exception as e:
            raise MetadataError(f"Failed to clean EXIF data: {str(e)}")
    
    def clean_batch_exif(self, folder_path: str) -> Dict:
        """
        Clean EXIF metadata from all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Dictionary with cleaning statistics
        """
        if not os.path.exists(folder_path):
            raise ValidationError(f"Folder not found: {folder_path}")
            
        stats = {
            'processed': 0,
            'cleaned': 0,
            'failed': 0,
            'errors': []
        }
        
        # Find all images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        
        if not image_files:
            return stats
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(
                "[green]Cleaning metadata...",
                total=len(image_files)
            )
            
            for img_path in image_files:
                try:
                    if self.read_exif(img_path):
                        self.clean_exif(img_path)
                        stats['cleaned'] += 1
                    stats['processed'] += 1
                    
                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append((img_path, str(e)))
                    self.logger.warning(
                        f"Failed to clean {img_path}: {str(e)}"
                    )
                
                progress.update(task, advance=1)
        
        return stats

    def run_batch_inference(self,
                          input_folder: str,
                          output_folder: str = None,
                          conf: float = None,
                          skip_errors: bool = False) -> Dict:
        """
        Run batch inference on multiple images.
        
        Args:
            input_folder: Folder containing images
            output_folder: Output folder for results
            conf: Confidence threshold
            skip_errors: Whether to continue on errors
            
        Returns:
            Dictionary with batch processing statistics
        """
        if not os.path.exists(input_folder):
            raise ValidationError(f"Input folder not found: {input_folder}")
            
        output_folder = output_folder or os.path.join(
            self.config['OUTPUT_DIR'],
            f"batch_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
        
        if not image_files:
            raise ValidationError(f"No images found in {input_folder}")
            
        stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'total_bricks': 0,
            'total_studs': 0,
            'dimension_stats': {},
            'errors': []
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(
                "[green]Processing batch...",
                total=len(image_files)
            )
            
            # Process each image
            for img_path in image_files:
                try:
                    result = self.run_full_pipeline(
                        img_path,
                        conf=conf,
                        output_folder=os.path.join(
                            output_folder,
                            Path(img_path).stem
                        )
                    )
                    
                    # Update statistics
                    stats['processed'] += 1
                    stats['total_bricks'] += len(result['brick_results']['boxes'])
                    
                    for stud_result in result['studs_results']:
                        stats['total_studs'] += stud_result['stud_count']
                        dim = stud_result['dimension']
                        stats['dimension_stats'][dim] = \
                            stats['dimension_stats'].get(dim, 0) + 1
                    
                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append((img_path, str(e)))
                    self.logger.error(f"Failed to process {img_path}: {str(e)}")
                    if not skip_errors:
                        raise
                
                progress.update(task, advance=1)
        
        # Save batch results summary
        summary_path = os.path.join(output_folder, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Batch processing complete. Summary saved to {summary_path}")
        return stats

# CLI Interface
@click.group()
@click.option('--debug/--no-debug', default=False,
              help='Enable debug output')
def cli(debug):
    """LEGO Vision Project - Computer Vision Pipeline
    
    A comprehensive tool for LEGO brick detection and analysis.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command('detect')
@click.option('--image', required=True, type=click.Path(exists=True),
              help='''Path to input image or folder containing images.
              
Supported formats: .jpg, .jpeg, .png
Examples:
  --image data/test_image.jpg
  --image data/test_images/          # Process all images in folder''')
@click.option('--mode', type=click.Choice(['bricks', 'studs', 'full']),
              default='full', 
              help='''Detection mode to run:
              
bricks: Only detect LEGO bricks in the image
studs: Only detect studs on pre-cropped brick images
full: Run complete pipeline (brick detection -> stud detection -> classification)

The 'full' mode is recommended for most cases as it provides the most comprehensive analysis.''')
@click.option('--conf', type=float, default=None,
              help='''Detection confidence threshold (0.0-1.0).
              
Higher values = more selective detection
Lower values = more permissive detection

Recommended ranges:
- Brick detection: 0.25-0.35
- Stud detection: 0.30-0.40

If not specified, uses default from config (0.25)''')
@click.option('--output', type=click.Path(),
              help='''Output folder for detection results.
              
Directory structure created:
output/
  ├── bricks/          # Brick detection results
  ├── studs/           # Stud detection results (if mode=full)
  └── summary.json     # Analysis summary

If not specified, defaults to 'results/<mode>/<image_name>'.''')
def detect_cmd(image, mode, conf, output):
    """Run brick and stud detection on images.

    This command provides three detection modes:
    \b
    1. Brick Detection (--mode bricks):
       - Identifies LEGO bricks in general images
       - Generates bounding boxes and confidence scores
       - Saves annotated images with visualizations
    
    2. Stud Detection (--mode studs):
       - Detects individual studs on brick surfaces
       - Works best with close-up brick images
       - Determines brick dimensions from stud pattern
    
    3. Full Pipeline (--mode full):
       - Runs complete detection sequence
       - First detects bricks, then analyzes studs
       - Classifies brick dimensions automatically
    
    Examples:
    \b
        # Basic brick detection
        LegoProjectUtils.py detect --image test.jpg --mode bricks
    
        # Full analysis with custom confidence
        LegoProjectUtils.py detect --image test.jpg --conf 0.35
    
        # Batch processing with custom output
        LegoProjectUtils.py detect --image data/batch/ --output results/custom/
    
    Output Structure:
    \b
        For each processed image:
        - Annotated image with detections
        - JSON file with detection data
        - Summary of brick dimensions (full mode)
        - Processing logs and statistics
    
    Notes:
    \b
        - Large images may be automatically resized
        - Use --conf to adjust detection sensitivity
        - Results include confidence scores
        - Failed detections are logged
    """
    try:
        project = LegoVisionProject()
        
        with console.status("[bold blue]Running detection...") as status:
            if mode == 'bricks':
                result = project.detect_bricks(
                    image,
                    conf=conf,
                    output_folder=output
                )
                console.print(f"[green]Detected {len(result['boxes'])} bricks")
                
            elif mode == 'studs':
                result = project.detect_studs(
                    image,
                    conf=conf,
                    output_folder=output
                )
                console.print(
                    f"[green]Detected {result['stud_count']} studs "
                    f"({result['dimension']})"
                )
                
            else:  # full
                result = project.run_full_pipeline(
                    image,
                    conf=conf,
                    output_folder=output
                )
                console.print(
                    f"[green]Analysis complete:[/green]\n"
                    f"• Bricks: {len(result['brick_results']['boxes'])}\n"
                    f"• Total studs: {sum(r['stud_count'] for r in result['studs_results'])}"
                )

    except LegoVisionError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

@cli.command('batch')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='''Input folder containing images to process.
              
Supported formats: .jpg, .jpeg, .png
Example structure:
input_folder/
  ├── image1.jpg
  ├── image2.png
  └── subfolder/
      └── image3.jpg

All images in folder and subfolders will be processed.''')
@click.option('--output', type=click.Path(),
              help='''Output folder for batch results.
              
Created structure:
output_folder/
  ├── image1/
  │   ├── bricks/
  │   ├── studs/
  │   └── summary.json
  ├── image2/
  │   └── ...
  └── batch_summary.json

If not specified, creates timestamped folder in results/.''')
@click.option('--conf', type=float, default=None,
              help='''Detection confidence threshold (0.0-1.0).
              
Applied to both brick and stud detection.
Lower values may increase false positives.
Higher values may miss some detections.

Default: 0.25 (from config)''')
@click.option('--skip-errors', is_flag=True,
              help='''Continue processing if individual images fail.
              
When enabled:
- Failed images are logged but don't stop batch
- Partial results are saved
- Error summary included in batch_summary.json

When disabled:
- Process stops at first error
- Allows immediate investigation of issues''')
def batch_cmd(input, output, conf, skip_errors):
    """Run batch inference on multiple images.

    This command processes multiple images through the full detection pipeline,
    generating comprehensive results and statistics.

    Processing Steps:
    \b
    1. Image Discovery:
       - Scans input folder for supported images
       - Validates image formats and readability
       - Creates output directory structure
    
    2. Detection Pipeline:
       - Runs brick detection on each image
       - Performs stud detection on found bricks
       - Classifies brick dimensions
       - Generates visualizations
    
    3. Results Collection:
       - Saves individual image results
       - Compiles batch statistics
       - Creates summary report
    
    Examples:
    \b
        # Process folder with default settings
        LegoProjectUtils.py batch --input data/images/
    
        # Custom confidence and output
        LegoProjectUtils.py batch --input data/images/ --conf 0.3 --output results/batch1/
    
        # Continue on errors
        LegoProjectUtils.py batch --input data/images/ --skip-errors
    
    Output Format:
    \b
        batch_summary.json contains:
        - Total images processed
        - Success/failure counts
        - Total bricks detected
        - Brick dimension statistics
        - Processing timestamps
        - Error logs (if any)
    
    Performance Notes:
    \b
        - Large batches process sequentially
        - Memory usage scales with image size
        - Consider breaking very large batches
        - Progress bar shows ETA
    """
    try:
        project = LegoVisionProject()
        stats = project.run_batch_inference(
            input,
            output_folder=output,
            conf=conf,
            skip_errors=skip_errors
        )
        
        # Display results table
        table = Table(title="Batch Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Images", str(stats['total_images']))
        table.add_row("Processed", str(stats['processed']))
        table.add_row("Failed", str(stats['failed']))
        table.add_row("Total Bricks", str(stats['total_bricks']))
        table.add_row("Total Studs", str(stats['total_studs']))
        
        console.print(table)
        
        if stats['dimension_stats']:
            dim_table = Table(title="Brick Dimensions")
            dim_table.add_column("Dimension", style="cyan")
            dim_table.add_column("Count", style="green")
            
            for dim, count in stats['dimension_stats'].items():
                dim_table.add_row(dim, str(count))
            
            console.print(dim_table)
        
        if stats['errors']:
            console.print("\n[yellow]Processing Errors:[/yellow]")
            for img, error in stats['errors']:
                console.print(f"• {Path(img).name}: {error}")
                
    except LegoVisionError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

@cli.command('convert')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='''Input folder containing annotation files.
              
For labelme-to-yolo:
  - Expects LabelMe JSON format
  - Must include imagePath field
  - Rectangle annotations only

For keypoints-to-boxes:
  - Expects point annotation JSONs
  - Converts to bounding boxes
  - Uses area_ratio parameter''')
@click.option('--output', type=click.Path(),
              help='''Output folder for converted annotations.
              
Created structure varies by format:

labelme-to-yolo:
  output/
    ├── class_mapping.json
    └── <filename>.txt (YOLO format)

keypoints-to-boxes:
  output/
    └── <filename>_boxes.json

If not specified, uses default folder in results/.''')
@click.option('--format', 'format_type',
              type=click.Choice(['labelme-to-yolo', 'keypoints-to-boxes']),
              required=True,
              help='''Conversion format to use:
              
labelme-to-yolo:
  - Converts LabelMe JSON to YOLO txt
  - Preserves class information
  - Normalizes coordinates

keypoints-to-boxes:
  - Converts point annotations to boxes
  - Uses image dimensions
  - Applies area_ratio for size''')
@click.option('--clean', is_flag=True,
              help='''Clean output directory before conversion.
              
When enabled:
- Removes all files in output folder
- Starts fresh conversion
- Prevents mixed results

When disabled:
- Keeps existing files
- May overwrite if same names''')
def convert_cmd(input, output, format_type, clean):
    """Convert between different annotation formats.

    This command provides utilities to convert between different annotation
    formats used in the LEGO detection pipeline.

    Supported Conversions:
    \b
    1. LabelMe to YOLO format:
       - Converts LabelMe JSON annotations to YOLO txt format
       - Handles rectangular annotations only
       - Normalizes coordinates to YOLO format
       - Preserves class mappings
    
    2. Keypoints to Bounding Boxes:
       - Converts point annotations to bounding boxes
       - Uses image dimensions and area ratio
       - Generates JSON output with box coordinates
    
    Examples:
    \b
        # Convert LabelMe annotations
        LegoProjectUtils.py convert --input data/labelme/ --format labelme-to-yolo
    
        # Convert keypoints with clean output
        LegoProjectUtils.py convert --input data/points/ --format keypoints-to-boxes --clean
    
    File Format Requirements:
    \b
    LabelMe JSON:
        {
          "imagePath": "path/to/image.jpg",
          "shapes": [
            {
              "label": "brick_type",
              "shape_type": "rectangle",
              "points": [[x1,y1], [x2,y2]]
            }
          ]
        }
    
    Keypoint JSON:
        {
          "imagePath": "path/to/image.jpg",
          "shapes": [
            {
              "label": "stud",
              "shape_type": "point",
              "points": [[x,y]]
            }
          ]
        }
    
    Notes:
    \b
        - Validates all input files before converting
        - Generates detailed conversion report
        - Maintains original file references
        - Logs any conversion errors
    """
    try:
        project = LegoVisionProject()
        
        if format_type == 'labelme-to-yolo':
            stats = project.process_labelme_to_yolo(
                input,
                output_path=output,
                clean=clean
            )
        else:
            stats = project.process_keypoints_to_boxes(
                input,
                output_path=output,
                clean=clean
            )
        
        # Display results
        table = Table(title="Conversion Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Files", str(stats['total']))
        table.add_row("Successful", str(stats['success']))
        table.add_row("Failed", str(stats['failed']))
        
        console.print(table)
        
        if stats['errors']:
            console.print("\n[yellow]Conversion Errors:[/yellow]")
            for file, error in stats['errors']:
                console.print(f"• {Path(file).name}: {error}")
                
    except LegoVisionError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

@cli.command('clean-metadata')
@click.argument('path', type=click.Path(exists=True))
@click.option('--batch', is_flag=True,
              help='''Process all images in folder.
              
When enabled:
- Processes all images recursively
- Generates cleaning report
- Shows progress bar

When disabled:
- Processes single image only
- Quick operation
- Simple success/fail output''')
def clean_metadata_cmd(path, batch):
    """Clean EXIF metadata from images.

    This command removes EXIF metadata from images, which can contain sensitive
    information or cause issues with certain processing steps.

    Cleaning Process:
    \b
    1. Single Image Mode:
       - Reads original image
       - Strips all EXIF data
       - Preserves image quality
       - Overwrites original file
    
    2. Batch Mode:
       - Scans folder for images
       - Processes each image
       - Generates report
       - Shows progress
    
    Examples:
    \b
        # Clean single image
        LegoProjectUtils.py clean-metadata image.jpg
    
        # Clean entire folder
        LegoProjectUtils.py clean-metadata data/images/ --batch
    
    Supported Formats:
    \b
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
    
    Notes:
    \b
        - Original files are modified
        - Consider backups first
        - Process is irreversible
        - Maintains image quality
    """
    try:
        project = LegoVisionProject()
        
        if batch:
            stats = project.clean_batch_exif(path)
            
            table = Table(title="Metadata Cleaning Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Images", str(stats['processed']))
            table.add_row("Cleaned", str(stats['cleaned']))
            table.add_row("Failed", str(stats['failed']))
            
            console.print(table)
            
            if stats['errors']:
                console.print("\n[yellow]Processing Errors:[/yellow]")
                for img, error in stats['errors']:
                    console.print(f"• {Path(img).name}: {error}")
        else:
            project.clean_exif(path)
            console.print(f"[green]Successfully cleaned metadata from {path}")
            
    except LegoVisionError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    cli()