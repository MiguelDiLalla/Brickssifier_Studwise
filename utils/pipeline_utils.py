"""
Pipeline Utilities for LEGO Bricks ML Vision

This module coordinates the full detection and classification pipeline,
integrating all other utility modules together.

Key features:
  - Full algorithm workflow coordination
  - Multi-step processing pipeline
  - Result aggregation and formatting
  - Batch processing capabilities

Author: Miguel DiLalla
"""

import os
import logging
import datetime
import json
import cv2
import numpy as np
from typing import Dict, List, Union, Optional

# Set up logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Pipeline Utils module loaded.")

# Import project modules
from utils.config_utils import config
from utils.detection_utils import detect_bricks, detect_studs
from utils.visualization_utils import create_composite_image
from utils.exif_utils import clean_exif_metadata

# Import rich utilities if available
try:
    from utils.rich_utils import (
        RICH_AVAILABLE, console, create_progress, 
        create_status_panel
    )
except ImportError:
    # Fallback if rich_utils is not available
    RICH_AVAILABLE = False
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        RICH_AVAILABLE = True
        console = Console()
    except ImportError:
        RICH_AVAILABLE = False
        print("Warning: 'rich' package not available. Install with: pip install rich")

def run_full_algorithm(image_path, save_annotated=False, output_folder="", force_rerun=False, logo=None, external_progress=None):
    """
    Runs the full inference pipeline: brick detection, stud detection, and classification.
    
    Args:
        image_path (str): Path to the image file to process
        save_annotated (bool): Whether to save annotated images to disk
        output_folder (str): Path to save results (created if not exists)
        force_rerun (bool): Force re-running detection even if cached results exist
        logo (np.ndarray, optional): Logo image to add to the bottom of output
        external_progress: Optional external progress context from CLI
        
    Returns:
        dict: Dictionary containing:
            - brick_results: Results from brick detection
            - studs_results: Results from stud detection for each brick
            - composite_image: Final annotated image with all results
            
    Notes:
        - Creates a comprehensive composite image showing all stages
        - Processes each detected brick independently for stud detection
        - Handles cases with multiple bricks in a single image
        - Supports external progress tracking for CLI integration
        - Can use cached results if force_rerun is False
    """
    # Handle empty output folder
    if output_folder == "" and save_annotated:
        output_folder = os.path.join(os.getcwd(), "results")
        os.makedirs(output_folder, exist_ok=True)
        logger.info("📂 Output folder not provided. Saving results in: %s", output_folder)

    # If force_rerun is True, clean existing EXIF metadata
    if force_rerun and isinstance(image_path, str):
        logger.info("🔄 Force rerun requested. Cleaning existing EXIF metadata.")
        clean_exif_metadata(image_path)

    # Use default logo from config if none provided
    if logo is None:
        logo = config.get("LOGO_NUMPY")

    # Step 1: Brick detection
    if RICH_AVAILABLE and not external_progress:
        with create_progress() as progress:
            task_bricks = progress.add_task("[green]Detecting bricks...", total=100)
            brick_results = detect_bricks(
                image_path,
                save_annotated=False,
                save_json=False,
                output_folder=output_folder,
                use_progress=False,
                force_rerun=force_rerun
            )
            progress.update(task_bricks, completed=100)
    else:
        logger.info("🧱 Running brick detection...")
        brick_results = detect_bricks(
            image_path,
            save_annotated=False,
            save_json=False,
            output_folder=output_folder,
            use_progress=False,
            force_rerun=force_rerun
        )

    if brick_results is None:
        logger.error("❌ Error during brick detection.")
        return None

    # Get cropped detections from brick detection
    cropped_detections = brick_results.get("cropped_detections", [])
    logger.info("🔍 Found %d brick detections.", len(cropped_detections))
    
    # Step 2: Process each detected brick for stud detection
    studs_results = []
    
    if RICH_AVAILABLE and not external_progress and len(cropped_detections) > 0:
        with create_progress() as progress:
            task_studs = progress.add_task(
                f"[blue]Detecting studs on {len(cropped_detections)} brick(s)...", 
                total=len(cropped_detections)
            )
            
            for idx, crop in enumerate(cropped_detections):
                logger.info("🔍 Processing brick %d of %d", idx + 1, len(cropped_detections))
                # Use numpy arrays directly for crops (no caching)
                studs_result = detect_studs(crop, save_annotated=False, force_rerun=True)
                if studs_result is None:
                    logger.error("❌ No studs detected in brick %d", idx + 1)
                    continue
                studs_results.append(studs_result)
                progress.update(task_studs, advance=1)
    else:
        # Process without rich progress display
        for idx, crop in enumerate(cropped_detections):
            logger.info("🔍 Processing brick %d of %d", idx + 1, len(cropped_detections))
            studs_result = detect_studs(crop, save_annotated=False, force_rerun=True)
            if studs_result is None:
                logger.error("❌ No studs detected in brick %d", idx + 1)
                continue
            studs_results.append(studs_result)

    # Handle case with no brick detections
    if len(cropped_detections) == 0:
        logger.warning("⚠️ No bricks detected. Running stud detection on the whole image...")
        studs_result = detect_studs(image_path, save_annotated=False, force_rerun=force_rerun)
        if studs_result is not None:
            studs_results = [studs_result]

    # Check if we got any valid stud detection results
    if not studs_results:
        logger.warning("⚠️ No valid stud detection results obtained.")
        
        # Still return brick detection results
        if save_annotated and output_folder and brick_results.get("annotated_image") is not None:
            output_path = os.path.join(output_folder, "bricks_only.jpg")
            cv2.imwrite(output_path, brick_results.get("annotated_image"))
            logger.info("💾 Saved brick detection image to: %s", output_path)
            
        return {
            "brick_results": brick_results,
            "studs_results": [],
            "composite_image": brick_results.get("annotated_image")
        }

    # Step 3: Create the composite visualization
    logger.info("🖼️ Creating composite visualization...")
    base_image = brick_results.get("annotated_image")
    stud_images = [result.get("annotated_image") for result in studs_results if result.get("annotated_image") is not None]
    
    composite_image = create_composite_image(base_image, stud_images, logo)
    
    # Save annotated image if requested
    if save_annotated and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        annotated_path = os.path.join(output_folder, "full_analysis.jpg")
        cv2.imwrite(annotated_path, composite_image)
        logger.info("💾 Full analysis image saved at: %s", annotated_path)
        
        # Optional: Save metadata
        metadata = {
            "brick_detection": {
                "count": len(cropped_detections),
                "timestamp": datetime.datetime.now().isoformat()
            },
            "stud_detection": [
                {
                    "index": idx,
                    "dimension": result.get("dimension", "Unknown"),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                for idx, result in enumerate(studs_results)
            ]
        }
        
        metadata_path = os.path.join(output_folder, "full_analysis_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info("💾 Analysis metadata saved at: %s", metadata_path)

    # Return the results dictionary
    return {
        "brick_results": brick_results,
        "studs_results": studs_results,
        "composite_image": composite_image
    }

def batch_process(image_paths, output_folder="batch_results", force_rerun=False):
    """
    Process multiple images in batch mode.
    
    Args:
        image_paths (List[str]): List of image paths to process
        output_folder (str): Base folder for saving results
        force_rerun (bool): Whether to force re-detection even if cached results exist
        
    Returns:
        Dict[str, Dict]: Dictionary mapping image paths to their result dictionaries
    """
    results = {}
    
    if not image_paths:
        logger.error("❌ No images provided for batch processing.")
        return results
    
    # Create the base output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    if RICH_AVAILABLE:
        with create_progress() as progress:
            task = progress.add_task(
                f"[green]Processing {len(image_paths)} images...", 
                total=len(image_paths)
            )
            
            for idx, image_path in enumerate(image_paths):
                # Create a specific subfolder for this image
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                image_output_folder = os.path.join(output_folder, image_name)
                os.makedirs(image_output_folder, exist_ok=True)
                
                logger.info("🔄 Processing image %d/%d: %s", 
                           idx + 1, len(image_paths), image_path)
                
                try:
                    result = run_full_algorithm(
                        image_path, 
                        save_annotated=True,
                        output_folder=image_output_folder,
                        force_rerun=force_rerun
                    )
                    results[image_path] = result
                except Exception as e:
                    logger.error("❌ Error processing image %s: %s", image_path, e)
                    results[image_path] = {"error": str(e)}
                
                progress.update(task, advance=1)
    else:
        # Fallback without rich progress display
        for idx, image_path in enumerate(image_paths):
            # Create a specific subfolder for this image
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_output_folder = os.path.join(output_folder, image_name)
            os.makedirs(image_output_folder, exist_ok=True)
            
            logger.info("🔄 Processing image %d/%d: %s", 
                       idx + 1, len(image_paths), image_path)
            
            try:
                result = run_full_algorithm(
                    image_path, 
                    save_annotated=True,
                    output_folder=image_output_folder,
                    force_rerun=force_rerun
                )
                results[image_path] = result
            except Exception as e:
                logger.error("❌ Error processing image %s: %s", image_path, e)
                results[image_path] = {"error": str(e)}
    
    # Create a summary report
    summary = {
        "total_images": len(image_paths),
        "successful": sum(1 for result in results.values() if "error" not in result),
        "failed": sum(1 for result in results.values() if "error" in result),
        "timestamp": datetime.datetime.now().isoformat(),
        "per_image": {
            path: {
                "status": "success" if "error" not in result else "error",
                "error_message": result.get("error", ""),
                "bricks_detected": len(result.get("brick_results", {}).get("cropped_detections", [])) if "error" not in result else 0,
                "dimensions": [r.get("dimension", "Unknown") for r in result.get("studs_results", [])] if "error" not in result else []
            }
            for path, result in results.items()
        }
    }
    
    # Save summary report
    summary_path = os.path.join(output_folder, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info("📊 Batch processing summary saved to: %s", summary_path)
    
    return results
