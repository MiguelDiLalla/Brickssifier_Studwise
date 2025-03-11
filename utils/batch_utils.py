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
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

from utils.config_utils import config
from utils.detection_utils import detect_bricks, detect_studs
from utils.classification_utils import (
    classify_dimensions,
    predict_dimension_from_pattern
)
from utils.exif_utils import clean_exif_metadata

logger = logging.getLogger(__name__)

def check_existing_annotations(image_paths: List[Path], output_folder: Path) -> Tuple[List[Path], List[Path]]:
    """Check which images already have annotation files.
    
    Args:
        image_paths: List of image file paths
        output_folder: Path to annotation output folder
        
    Returns:
        Tuple of (to_process, already_processed) image paths
    """
    to_process = []
    already_processed = []
    
    for img_path in image_paths:
        annotation_path = output_folder / f"{img_path.stem}.txt"
        if annotation_path.exists():
            already_processed.append(img_path)
        else:
            to_process.append(img_path)
            
    return to_process, already_processed

def process_batch_inference(
    input_folder, 
    output_folder, 
    dimensions_map, 
    progress=None, 
    image_paths=None,
    skip_errors=False,
    progress_table=None
):
    """Process a batch of images for inference.
    
    Args:
        input_folder (str): Input folder path
        output_folder (str): Output folder path
        dimensions_map (dict): Mapping of class IDs to dimensions
        progress (Progress, optional): Rich progress instance
        image_paths (list, optional): Pre-validated list of image paths
        skip_errors (bool): Whether to continue processing on errors
        progress_table (Table, optional): Rich table instance for progress display
        
    Returns:
        dict: Processing statistics
    """
    stats = {
        'processed_images': 0,
        'skipped_images': 0,  # Add counter for skipped images
        'total_bricks': 0,
        'valid_bricks': 0,
        'unknown_bricks': 0,
        'cleaned_images': 0,  # Added initialization
        'failed_cleans': 0,   # Track failed EXIF cleaning
        'class_distribution': {},  # Initialize empty distribution
        'errors': []
    }
    
    # Create output folder
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Use provided image paths or find images in folder
    if image_paths is None:
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(Path(input_folder).glob(ext))
    
    # Check for existing annotations
    to_process, already_processed = check_existing_annotations(image_paths, output_folder)
    stats['skipped_images'] = len(already_processed)
    
    # Initialize Rich layout with safe table structure
    layout = Layout()
    layout.split(
        Layout(Panel.fit("[bold blue]Batch Processing Progress[/bold blue]", border_style="blue"), size=3),
        Layout(name="main")
    )
    
    # Create and pre-populate stats table
    stats_table = Table.grid(padding=(0,1))
    stats_table.add_column("Metric", style="cyan", width=20)
    stats_table.add_column("Value", style="green")
    
    # Add initial rows to prevent index errors
    stats_table.add_row("Status", "Initializing...")
    stats_table.add_row("Progress", "0/0")
    stats_table.add_row("Processing", "")
    stats_table.add_row("Valid Bricks", "0")
    stats_table.add_row("Skipped", str(stats['skipped_images']))
    stats_table.add_row("Errors", "0")
    
    layout["main"].update(stats_table)

    total_images = len(to_process)
    if total_images == 0:
        logger.info(f"No new images to process. {stats['skipped_images']} images already processed.")
        return stats

    with Live(layout, refresh_per_second=4, transient=True) as live:
        for img_path in to_process:
            try:
                # Update progress display safely
                current_file = Path(img_path).name
                stats_table.rows[0] = ("Status", f"Processing {current_file}")
                stats_table.rows[1] = ("Progress", f"{stats['processed_images']}/{total_images}")
                stats_table.rows[2] = ("Processing", str(img_path.name))
                stats_table.rows[3] = ("Valid Bricks", str(stats['valid_bricks']))
                stats_table.rows[4] = ("Skipped", str(stats['skipped_images']))
                if stats['errors']:
                    stats_table.rows[5] = ("Errors", f"[red]{len(stats['errors'])}[/red]")
                
                layout["main"].update(stats_table)

                # Clean EXIF metadata with error handling
                try:
                    clean_exif_metadata(str(img_path))
                    stats['cleaned_images'] += 1
                except Exception as e:
                    stats['failed_cleans'] += 1
                    if not skip_errors:
                        raise
                    stats['errors'].append((str(img_path), f"EXIF cleaning failed: {str(e)}"))
                    continue

                # Process image
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
                
            except Exception as e:
                stats['errors'].append((str(img_path), str(e)))
                if not skip_errors:
                    raise
    
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
    # Clear previous output
    console.clear()
    
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