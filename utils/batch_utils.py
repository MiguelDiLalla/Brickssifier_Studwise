"""
Batch Processing Utilities for LEGO Bricks ML Vision

This module handles batch inference operations for generating
YOLO format annotations from brick detections.

Author: Miguel DiLalla
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.progress import Progress
from rich.console import Console
from rich.table import Table

from utils.config_utils import config
from utils.detection_utils import detect_bricks, detect_studs
from utils.classification_utils import (
    classify_dimensions,
    predict_dimension_from_pattern
)
from utils.exif_utils import clean_exif_metadata

logger = logging.getLogger(__name__)

def process_batch_inference(
    input_folder: str,
    output_folder: str,
    dimensions_map: Dict[int, str],
    progress: Optional[Progress] = None
) -> Dict:
    """
    Process a batch of images for brick detection and annotation generation.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to save YOLO annotations and metadata
        dimensions_map: Dictionary mapping class IDs to brick dimensions
        progress: Optional Rich progress instance for status updates
        
    Returns:
        dict: Processing statistics and results
    """
    # Initialize stats at the start
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'total_bricks': 0,
        'valid_bricks': 0,
        'unknown_bricks': 0,
        'class_distribution': {},
        'cleaned_images': 0
    }

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    if not image_files:
        logger.error("No image files found in input folder")
        return {'status': 'error', 'message': 'No images found'}

    stats['total_images'] = len(image_files)
    
    # Clean EXIF metadata from all images first
    if progress:
        progress.update(0, description="Cleaning EXIF metadata")
    
    for img_path in image_files:
        try:
            clean_exif_metadata(str(img_path))
            stats['cleaned_images'] += 1
        except Exception as e:
            logger.error(f"Failed to clean EXIF metadata from {img_path}: {e}")

    # Process each image
    for img_path in image_files:
        if progress:
            progress.update(0, description=f"Processing {img_path.name}")
        
        # Run detection without internal progress tracking
        brick_results = detect_bricks(
            str(img_path), 
            save_annotated=False,
            use_progress=False  # Disable internal progress tracking
        )
        
        if not brick_results or 'boxes' not in brick_results:
            continue
            
        boxes = brick_results['boxes']
        valid_detections = []
        
        # Process each brick detection
        for box in boxes:
            cropped_brick = brick_results['orig_image'][
                int(box[1]):int(box[3]), 
                int(box[0]):int(box[2])
            ]
            
            # Use the classification utils directly
            stud_results = detect_studs(cropped_brick, save_annotated=False)
            if stud_results and "dimension" in stud_results:
                dimension = stud_results["dimension"]
                # Get class ID from dimension mapping
                class_id = None
                for id, dim in dimensions_map.items():
                    if dim == dimension:
                        class_id = id
                        break
                    
                if class_id is not None:
                    # Convert to YOLO format
                    img_height, img_width = brick_results['orig_image'].shape[:2]
                    x_center = ((box[0] + box[2]) / 2) / img_width
                    y_center = ((box[1] + box[3]) / 2) / img_height
                    width = (box[2] - box[0]) / img_width
                    height = (box[3] - box[1]) / img_height
                    
                    valid_detections.append(f"{class_id} {x_center} {y_center} {width} {height}")
                    stats['valid_bricks'] += 1
                    stats['class_distribution'][dimension] = \
                        stats['class_distribution'].get(dimension, 0) + 1
        
        # Save YOLO annotation file if we have valid detections
        if valid_detections:
            yolo_path = Path(output_folder) / f"{img_path.stem}.txt"
            with open(yolo_path, 'w') as f:
                f.write('\n'.join(valid_detections))
        
        stats['processed_images'] += 1
        stats['total_bricks'] += len(boxes)
        
        if progress:
            progress.update(0, advance=1)
    
    # Save metadata
    metadata = {
        'statistics': stats,
        'config': {
            'input_folder': str(input_folder),
            'output_folder': str(output_folder),
            'end_time': datetime.now().isoformat(),
            'classes': dimensions_map
        }
    }
    
    metadata_path = Path(output_folder) / 'batch_inference_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return stats

def display_batch_results(stats: Dict, console: Console):
    """Display formatted batch processing results."""
    # Processing summary
    summary = Table(title="Batch Processing Summary", show_header=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    
    summary.add_row("Total Images", str(stats['total_images']))
    summary.add_row("Processed Images", str(stats['processed_images']))
    summary.add_row("Total Bricks Detected", str(stats['total_bricks']))
    summary.add_row("Valid Bricks", str(stats['valid_bricks']))
    summary.add_row("Unknown/Invalid Bricks", str(stats['unknown_bricks']))
    
    console.print(summary)
    
    # Class distribution
    if stats['class_distribution']:
        dist_table = Table(title="Dimension Distribution", show_header=True)
        dist_table.add_column("Dimension", style="cyan")
        dist_table.add_column("Count", style="green")
        dist_table.add_column("Percentage", style="yellow")
        
        total = sum(stats['class_distribution'].values())
        for dim, count in stats['class_distribution'].items():
            percentage = (count / total) * 100
            dist_table.add_row(
                dim,
                str(count),
                f"{percentage:.1f}%"
            )
        
        console.print(dist_table)