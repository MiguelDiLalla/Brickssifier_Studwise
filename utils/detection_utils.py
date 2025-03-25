"""
Detection Utilities for LEGO Bricks ML Vision

This module handles brick and stud detection operations using YOLO models

Key features:
  - Brick detection with YOLO models
  - Stud detection within brick regions
  - Model inference and result processing
  - Detection metadata generation

Author: Miguel DiLalla
"""

import logging
import os
import json
import datetime
import numpy as np
import cv2
from ultralytics import YOLO

# Set up logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Detection Utils module loaded.")

# Import project modules
from utils.config_utils import config
from utils.metadata_utils import extract_metadata_from_yolo_result
from utils.exif_utils import write_exif, read_exif
from utils.visualization_utils import read_detection  # Add this line

# Import rich utilities if available
try:
    from utils.rich_utils import (
        RICH_AVAILABLE, console, create_progress, 
        create_status_panel, display_results_table
    )
except ImportError:
    # Fallback if rich_utils is not available
    RICH_AVAILABLE = False
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        RICH_AVAILABLE = True
        console = Console()
    except ImportError:
        RICH_AVAILABLE = False
        print("Warning: 'rich' package not available. Install with: pip install rich")

# Import classification functions directly
from utils.classification_utils import (
    classify_dimensions,
    analyze_stud_pattern,
    predict_dimension_from_pattern
)

def detect_bricks(image_input, model=None, conf=0.25, save_json=False, save_annotated=False, output_folder="", use_progress=True, force_rerun=False):
    """
    Performs brick detection using the provided YOLO model with rich progress display.
    
    Args:
        image_input (Union[str, np.ndarray]): Either image path (str) or numpy array
        model (YOLO, optional): YOLO model instance (uses default from config if None)
        conf (float): Confidence threshold for detections (0.0-1.0)
        save_json (bool): If True, saves detection metadata as JSON
        save_annotated (bool): If True, saves annotated image
        output_folder (str): Directory to save outputs to
        use_progress (bool): Whether to use progress display
        force_rerun (bool): If True, forces re-running detection even if cached results exist
        
    Returns:
        dict: Dictionary containing:
            - orig_image: Original image
            - annotated_image: Image with annotations
            - cropped_detections: List of cropped regions
            - metadata: Complete metadata dictionary
            - boxes: Detected bounding boxes
            
    Notes:
        - Uses either provided model or loads default from config
        - Shows real-time progress with rich display if available
        - Saves outputs based on specified flags
        - Can retrieve cached results from EXIF if available and force_rerun is False
    """
    # Define results at the beginning to ensure it exists
    results = None
    
    # Check for cached detection if image_input is a file path and not forcing rerun
    if isinstance(image_input, str) and not force_rerun:
        cached_results = read_detection(image_input)
        if cached_results and cached_results.get("status") == "success":
            if "mode" in cached_results.get("metadata", {}) and cached_results["metadata"]["mode"] == "brick":
                logger.info("📋 Using cached brick detection results from EXIF metadata.")
                
                # Return cached results but still save outputs if requested
                if output_folder and save_annotated:
                    annotated_image = cached_results.get("annotated_image")
                    annotated_path = os.path.join(output_folder, "cached_brick_detection.jpg")
                    os.makedirs(output_folder, exist_ok=True)
                    cv2.imwrite(annotated_path, annotated_image)
                    logger.info("💾 Cached annotated image saved at: %s", annotated_path)
                    
                return cached_results
    
    with console.status("[bold green]Loading image and model...") if RICH_AVAILABLE else nullcontext() as status:
        # Load model if not provided
        if model is None:
            model = config.get("LOADED_MODELS", {}).get("bricks")
            if model is None:
                logger.error("❌ No bricks model loaded.")
                return None
        
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                logger.error("❌ Failed to load image from path: %s", image_input)
                return None
        else:
            image = image_input
            
        if not output_folder:
            logger.warning("⚠️ No output folder provided. Results will not be saved.")
        else:
            os.makedirs(output_folder, exist_ok=True)
        
        if RICH_AVAILABLE and status:
            status.update("[bold green]Running detection...")
    
    if use_progress and RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold green]{task.completed}/{task.total}"),
            TimeElapsedColumn()
        ) as progress:
            detection_task = progress.add_task("[green]Detecting bricks...", total=100)
            
            # Run detection
            results = model.predict(source=image, conf=conf)
            progress.update(detection_task, advance=50)
            
            metadata = extract_metadata_from_yolo_result(results, image_input)
            # Set the mode explicitly to "brick" to distinguish the detection type
            metadata["mode"] = "brick"
            boxes_np = results[0].boxes.xyxy.cpu().numpy() if results and len(results) > 0 and results[0].boxes.xyxy is not None else np.array([])
            annotated_image = results[0].plot(labels=True) if boxes_np.size > 0 else image.copy()
            
            cropped_detections = []
            for box in boxes_np:
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                cropped_detections.append(crop)
            
            # Save outputs if requested and output_folder is provided
            if output_folder:
                if save_annotated:
                    # Save the annotated image
                    composite_path = os.path.join(output_folder, "brick_detection.jpg")
                    cv2.imwrite(composite_path, annotated_image)
                    metadata["annotated_image_path"] = composite_path
                    logger.info("💾 Saved annotated image to: %s", composite_path)
                    
                if save_json:
                    # Save metadata as JSON
                    json_path = os.path.join(output_folder, "brick_metadata.json")
                    with open(json_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    metadata["json_results_path"] = json_path
                    logger.info("💾 Saved metadata to: %s", json_path)
                    
            progress.update(detection_task, completed=100)
    else:
        # Run detection without progress display
        results = model.predict(source=image, conf=conf)
        metadata = extract_metadata_from_yolo_result(results, image_input)
        # Set the mode explicitly to "brick" to distinguish the detection type
        metadata["mode"] = "brick"
        boxes_np = results[0].boxes.xyxy.cpu().numpy() if results and len(results) > 0 and results[0].boxes.xyxy is not None else np.array([])
        annotated_image = results[0].plot(labels=True) if boxes_np.size > 0 else image.copy()
        
        cropped_detections = []
        for box in boxes_np:
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            cropped_detections.append(crop)
        
        # Save outputs if requested and output_folder is provided
        if output_folder:
            if save_annotated:
                # Save the annotated image
                composite_path = os.path.join(output_folder, "brick_detection.jpg")
                cv2.imwrite(composite_path, annotated_image)
                metadata["annotated_image_path"] = composite_path
                logger.info("💾 Saved annotated image to: %s", composite_path)
                
            if save_json:
                # Save metadata as JSON
                json_path = os.path.join(output_folder, "brick_metadata.json")
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                metadata["json_results_path"] = json_path
                logger.info("💾 Saved metadata to: %s", json_path)
    
    # Check results safely after it's been defined
    if results and len(results) > 0 and RICH_AVAILABLE:
        display_results_table("Brick Detection Results", [
            ("Bricks detected", str(len(boxes_np))),
            ("Confidence threshold", f"{conf:.2f}"),
            ("Processing time", f"{metadata['speed']['inference']:.3f}s")
        ])
    else:
        logger.info("📊 Detected %d bricks with confidence threshold %.2f in %.3fs", 
                   len(boxes_np), conf, metadata['speed']['inference'])
    
    try:
        if isinstance(image_input, str):
            # Always update EXIF for the original image file
            write_exif(image_input, metadata)
        
        # Additionally update EXIF for annotated images if we're saving them
        if save_annotated and metadata.get("annotated_image_path"):
            write_exif(metadata["annotated_image_path"], metadata)
    except Exception as e:
        logger.error("❌ Failed to write EXIF metadata: %s", e)
        
    return {
        "orig_image": image,
        "annotated_image": annotated_image,
        "cropped_detections": cropped_detections,
        "metadata": metadata,
        "boxes": boxes_np,
        "status": "success"  # Add status for consistency with read_detection
    }

def detect_studs(image_input, model=None, conf=0.25, save_annotated=False, output_folder="", force_rerun=False):
    """
    Detects studs on LEGO bricks and classifies dimensions.
    
    Args:
        image_input (Union[str, np.ndarray]): Either image path (str) or numpy array
        model (YOLO, optional): YOLO model instance (uses default from config if None)
        conf (float): Confidence threshold for detections (0.0-1.0)
        save_annotated (bool): If True, saves annotated image
        output_folder (str): Directory to save outputs to
        force_rerun (bool): If True, forces re-running detection even if cached results exist
        
    Returns:
        dict: Dictionary containing:
            - orig_image: Original image
            - annotated_image: Image with annotations
            - dimension: Classified brick dimension (e.g., "2x4")
            - metadata: Complete metadata dictionary
            - status: Processing status
            
    Notes:
        - Can retrieve cached results from EXIF if available and force_rerun is False
        - Integrates with classify_dimensions for dimension determination
        - Saves metadata to image EXIF data for future reference
    """
    # Remove circular import
    from utils.visualization_utils import read_detection
    
    # Message if output folder is not provided
    if not output_folder:
        logger.warning("⚠️ No output folder provided. Results will not be saved.")
    
    # Check for cached detection if image_input is a file path and not forcing rerun
    detection_results = None
    if isinstance(image_input, str) and not force_rerun:
        detection_results = read_detection(image_input)
        if detection_results and detection_results.get("status") == "success":
            if "mode" in detection_results.get("metadata", {}) and detection_results["metadata"]["mode"] == "stud":
                logger.info("📋 Using cached stud detection results from EXIF metadata.")
                
                # Return cached results but still save outputs if requested
                if output_folder and save_annotated:
                    annotated_image = detection_results.get("annotated_image")
                    annotated_path = os.path.join(output_folder, "cached_stud_detection.jpg")
                    os.makedirs(output_folder, exist_ok=True)
                    cv2.imwrite(annotated_path, annotated_image)
                    logger.info("💾 Cached annotated image saved at: %s", annotated_path)
                    
                return detection_results

    if model is None:
        model = config.get("LOADED_MODELS", {}).get("studs")
        if model is None:
            logger.error("❌ No studs model loaded.")
            return None

    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            logger.error("❌ Failed to load image from path: %s", image_input)
            return None
    else:
        image = image_input

    try:
        orig_image = image.copy()
        annotated_image = image.copy()
        results = model.predict(source=image, conf=conf)
        if isinstance(image_input, str):
            results[0].path = image_input
            
        # Use classification utils directly
        dimension_info = classify_dimensions(results, orig_image)
        dimension_result = dimension_info.get("dimension", "Unknown")
        annotated_image = dimension_info.get("annotated_image", annotated_image)
        
        metadata = extract_metadata_from_yolo_result(results, orig_image)
        metadata["mode"] = "stud"
        metadata["dimension"] = dimension_result
        
        boxes_np = (results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else np.array([]))
        if boxes_np.size == 0:
            logger.warning("⚠️ No detections found.")
            return {
                "orig_image": orig_image,
                "annotated_image": annotated_image,
                "dimension": "No studs detected",
                "metadata": {},
                "status": "no_detections"
            }
        
    except Exception as e:
        logger.error("❌ Error during studs detection: %s", e)
        return None

    # Create output folder if not provided
    if not output_folder:
        output_folder = os.path.join(os.getcwd(), "results", "studs")
        os.makedirs(output_folder, exist_ok=True)   

    try:
        if isinstance(image_input, str):
            # Always update EXIF for the original image file
            write_exif(image_input, metadata)
    except Exception as e:
        logger.error("❌ Failed to write EXIF metadata: %s", e)
    
    # Save annotated image if requested
    if save_annotated:
        # Create and save the annotated image
        annotated_path = os.path.join(output_folder, "stud_detection.jpg")
        cv2.imwrite(annotated_path, annotated_image)
        metadata["annotated_image_path"] = annotated_path
        logger.info("💾 Annotated image saved at: %s", annotated_path)
        
        # Also write EXIF to annotated image if we saved it
        try:
            write_exif(annotated_path, metadata)
        except Exception as e:
            logger.error("❌ Failed to write EXIF metadata to annotated image: %s", e)

    return {
        "orig_image": orig_image,
        "annotated_image": annotated_image,
        "dimension": metadata.get("dimension", "Unknown"),
        "metadata": metadata,
        "status": "success"
    }

def detect_multiclass(image_input, model=None, conf=0.25, save_json=False, save_annotated=False, output_folder="", use_progress=True, force_rerun=False):
    """
    Performs multiclass detection using the provided YOLO model with rich progress display.
    
    Args:
        image_input (Union[str, np.ndarray]): Either image path (str) or numpy array
        model (YOLO, optional): YOLO model instance (uses default from config if None)
        conf (float): Confidence threshold for detections (0.0-1.0)
        save_json (bool): If True, saves detection metadata as JSON
        save_annotated (bool): If True, saves annotated image
        output_folder (str): Directory to save outputs to
        use_progress (bool): Whether to use progress display
        force_rerun (bool): If True, forces re-running detection even if cached results exist
        
    Returns:
        dict: Dictionary containing:
            - orig_image: Original image
            - annotated_image: Image with annotations 
            - cropped_detections: List of cropped regions
            - metadata: Complete metadata dictionary
            - boxes: Detected bounding boxes
    """
    # Define results at the beginning to ensure it exists
    results = None
    
    # Check for cached detection if image_input is a file path and not forcing rerun
    if isinstance(image_input, str) and not force_rerun:
        cached_results = read_detection(image_input)
        if cached_results and cached_results.get("status") == "success":
            if "mode" in cached_results.get("metadata", {}) and cached_results["metadata"]["mode"] == "multiclass":
                logger.info("📋 Using cached multiclass detection results from EXIF metadata.")
                
                if output_folder and save_annotated:
                    annotated_image = cached_results.get("annotated_image")
                    annotated_path = os.path.join(output_folder, "cached_multiclass_detection.jpg")
                    os.makedirs(output_folder, exist_ok=True)
                    cv2.imwrite(annotated_path, annotated_image)
                    logger.info("💾 Cached annotated image saved at: %s", annotated_path)
                    
                return cached_results

    with console.status("[bold green]Loading image and model...") if RICH_AVAILABLE else nullcontext() as status:
        # Load model if not provided
      
        if model is None:
            model = config.get("LOADED_MODELS", {}).get("multiclass")
            if model is None:
                logger.error("❌ No multiclass model loaded.")
                return {
                    "status": "error",
                    "message": "No multiclass model loaded",
                    "orig_image": image,
                    "annotated_image": image.copy() if image is not None else None,
                    "cropped_detections": [],
                    "metadata": {},
                    "boxes": np.array([])
                }
        
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                logger.error("❌ Failed to load image from path: %s", image_input)
                return None
        else:
            image = image_input
            
        if not output_folder:
            logger.warning("⚠️ No output folder provided. Results will not be saved.")
        else:
            os.makedirs(output_folder, exist_ok=True)
        
        if RICH_AVAILABLE and status:
            status.update("[bold green]Running detection...")

    if use_progress and RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold green]{task.completed}/{task.total}"),
            TimeElapsedColumn()
        ) as progress:
            detection_task = progress.add_task("[green]Detecting bricks...", total=100)
            
            # Run detection
            results = model.predict(source=image, conf=conf)
            progress.update(detection_task, advance=50)
            
            metadata = extract_metadata_from_yolo_result(results, image_input)
            metadata["mode"] = "multiclass"

            # Extract class names dictionary from model
            try:
                class_names = model.names if hasattr(model, 'names') else {}
                logger.info("📋 Found %d class names in model", len(class_names))
            except Exception as e:
                logger.error("❌ Failed to extract class names from model: %s", e)
                class_names = {}

            # Enrich metadata by replacing class IDs with class names
            if "boxes_coordinates" in metadata and class_names:
                try:
                    replaced_count = 0
                    # Iterate through numeric string keys in boxes_coordinates
                    for _, box_info in metadata["boxes_coordinates"].items():
                        if "class" in box_info and box_info["class"] in class_names:
                            box_info["class"] = class_names[box_info["class"]]
                            replaced_count += 1
                    logger.info("✅ Replaced %d class IDs with class names", replaced_count)
                except Exception as e:
                    logger.error("❌ Error replacing class names in metadata: %s", e)
            else:
                logger.warning("⚠️ No boxes in metadata or no class names available")


            boxes_np = results[0].boxes.xyxy.cpu().numpy() if results and len(results) > 0 and results[0].boxes.xyxy is not None else np.array([])
            annotated_image = results[0].plot(labels=True) if boxes_np.size > 0 else image.copy()
            
            cropped_detections = []
            for box in boxes_np:
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                cropped_detections.append(crop)
            
            if output_folder:
                if save_annotated:
                    composite_path = os.path.join(output_folder, "multiclass_detection.jpg")
                    cv2.imwrite(composite_path, annotated_image)
                    metadata["annotated_image_path"] = composite_path
                    logger.info("💾 Saved annotated image to: %s", composite_path)
                    
                if save_json:
                    json_path = os.path.join(output_folder, "multiclass_metadata.json")
                    with open(json_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    metadata["json_results_path"] = json_path
                    logger.info("💾 Saved metadata to: %s", json_path)
                    
            progress.update(detection_task, completed=100)
    else:
        # Run detection without progress display
        results = model.predict(source=image, conf=conf)
        metadata = extract_metadata_from_yolo_result(results, image_input)
        metadata["mode"] = "multiclass"
        boxes_np = results[0].boxes.xyxy.cpu().numpy() if results and len(results) > 0 and results[0].boxes.xyxy is not None else np.array([])
        annotated_image = results[0].plot(labels=True) if boxes_np.size > 0 else image.copy()
        
        cropped_detections = []
        for box in boxes_np:
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            cropped_detections.append(crop)
        
        if output_folder:
            if save_annotated:
                composite_path = os.path.join(output_folder, "multiclass_detection.jpg")
                cv2.imwrite(composite_path, annotated_image)
                metadata["annotated_image_path"] = composite_path
                logger.info("💾 Saved annotated image to: %s", composite_path)
                
            if save_json:
                json_path = os.path.join(output_folder, "multiclass_metadata.json")
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                metadata["json_results_path"] = json_path
                logger.info("💾 Saved metadata to: %s", json_path)

    if results and len(results) > 0 and RICH_AVAILABLE:
        display_results_table("Multiclass Detection Results", [
            ("Objects detected", str(len(boxes_np))),
            ("Confidence threshold", f"{conf:.2f}"),
            ("Processing time", f"{metadata['speed']['inference']:.3f}s")
        ])
    else:
        logger.info("📊 Detected %d objects with confidence threshold %.2f in %.3fs", 
                   len(boxes_np), conf, metadata['speed']['inference'])
    
    try:
        if isinstance(image_input, str):
            write_exif(image_input, metadata)
        
        if save_annotated and metadata.get("annotated_image_path"):
            write_exif(metadata["annotated_image_path"], metadata)
    except Exception as e:
        logger.error("❌ Failed to write EXIF metadata: %s", e)
        
    return {
        "orig_image": image,
        "annotated_image": annotated_image,
        "cropped_detections": cropped_detections,
        "metadata": metadata,
        "boxes": boxes_np,
        "status": "success"
    }

# Context manager for non-rich environments
class nullcontext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def update(self, *args, **kwargs):
        pass

def display_results_table(title: str, results: list) -> None:
    """Display detection results in a formatted table.
    
    Args:
        title: Title for the results table
        results: List of detection results to display
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{title}")
    logger.info("-" * 50)
    for result in results:
        logger.info(result)
    logger.info("-" * 50)