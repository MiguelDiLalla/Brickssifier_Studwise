"""
Metadata Utilities for LEGO Bricks ML Vision

This module handles metadata extraction and processing from detection results,
providing structured information about detections and classifications.

Key features:
  - Detection metadata extraction from YOLO results
  - System information collection
  - Detection history tracking
  - Structured metadata formatting

Author: Miguel DiLalla
"""

import logging
import datetime
import platform
import os
import json
import numpy as np

# Set up logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Metadata Utils module loaded.")

# Import project modules
from utils.config_utils import config

def extract_metadata_from_yolo_result(results, orig_image):
    """
    Extracts relevant metadata from YOLO results and the original image.
    
    Args:
        results (list): YOLO detection results
        orig_image (Union[str, np.ndarray]): Either image path or numpy array
            
    Returns:
        dict: Structured metadata dictionary with detection results and system info
        
    Notes:
        - Handles image as numpy array or file path
        - Records detailed box information, confidence scores, and class IDs
        - Preserves previous scan information when available
    """
    # Get image dimensions
    if isinstance(orig_image, str):
        loaded_image = cv2.imread(orig_image)
        shape = list(loaded_image.shape[:2]) if loaded_image is not None else [0, 0]
    elif hasattr(orig_image, "shape"):
        shape = list(orig_image.shape[:2])
    else:
        shape = [0, 0]

    # Get image path if applicable
    image_path = results[0].path if hasattr(results[0], "path") and results[0].path else ""

    # Check for previous metadata if it's a file path
    from utils.exif_utils import read_exif
    previous_metadata = {}
    if isinstance(orig_image, str) and image_path:
        try:
            previous_metadata = read_exif(image_path)
        except Exception as e:
            logger.error("❌ Error reading EXIF from %s: %s", image_path, e)
            previous_metadata = {}
    
    # Get times scanned from previous metadata or initialize
    times_scanned = previous_metadata.get("TimesScanned", 0)
    times_scanned = times_scanned + 1 if times_scanned else 1

    # Extract detection details
    boxes_np = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else np.array([])
    confidences_np = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, "conf") and results[0].boxes.conf is not None else np.array([])
    classes_np = results[0].boxes.cls.cpu().numpy() if hasattr(results[0].boxes, "cls") and results[0].boxes.cls is not None else np.array([])

    # Build detailed information for each detection
    boxes = []
    for idx, box in enumerate(boxes_np):
        box_info = {
            "coordinates": box.tolist(),
            "confidence": float(confidences_np[idx]) if idx < len(confidences_np) else None,
            "class": int(classes_np[idx]) if idx < len(classes_np) else None
        }
        boxes.append(box_info)
    
    # Get processing speed information
    speed_data = results[0].speed if hasattr(results[0], "speed") else {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    
    # Get class names from results for mode identification
    class_names = results[0].names if hasattr(results[0], "names") else {}
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    
    # Determine mode based on class names
    mode_value = class_names[0] if class_names else "obj"

    # Construct the complete metadata dictionary
    metadata = {
        "boxes_coordinates": {str(idx): box for idx, box in enumerate(boxes)},
        "orig_shape": shape,
        "speed": {
            "preprocess": speed_data.get("preprocess", 0.0), 
            "inference": speed_data.get("inference", 0.0), 
            "postprocess": speed_data.get("postprocess", 0.0)
        },
        "mode": mode_value,
        "path": image_path,
        "os_full_version_name": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "timestamp": datetime.datetime.now().isoformat(),
        "annotated_image_path": "",
        "json_results_path": "",
        "TimesScanned": times_scanned,
        "Repository": config["REPO_URL"],
        "message": "LEGO Bricks ML Vision Detection Result"
    }
    
    return metadata

def format_detection_summary(metadata):
    """
    Creates a user-friendly summary from detection metadata.
    
    Args:
        metadata (dict): Metadata dictionary from extract_metadata_from_yolo_result
        
    Returns:
        str: Formatted summary text with detection details
    """
    mode = metadata.get("mode", "unknown")
    box_count = len(metadata.get("boxes_coordinates", {}))
    
    # Calculate average confidence
    confidences = []
    for _, box_info in metadata.get("boxes_coordinates", {}).items():
        if isinstance(box_info, dict) and "confidence" in box_info:
            confidences.append(box_info["confidence"])
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Format processing time
    speed_data = metadata.get("speed", {})
    total_time = sum(speed_data.values())
    
    # Create the summary
    summary_lines = [
        f"Detection Type: {mode}",
        f"Objects Detected: {box_count}",
        f"Average Confidence: {avg_confidence:.2f}",
        f"Processing Time: {total_time:.3f}s",
        f"Timestamp: {metadata.get('timestamp', 'unknown')}"
    ]
    
    # Add dimension info if available
    if "dimension" in metadata:
        summary_lines.insert(2, f"Dimension: {metadata['dimension']}")
    
    return "\n".join(summary_lines)

def export_metadata_json(metadata, output_path):
    """
    Exports metadata to a JSON file.
    
    Args:
        metadata (dict): Metadata dictionary
        output_path (str): Path to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info("💾 Metadata exported to %s", output_path)
        return True
    except Exception as e:
        logger.error("❌ Failed to export metadata: %s", e)
        return False

def calculate_detection_metrics(metadata_list):
    """
    Calculates aggregate metrics from multiple detection results.
    
    Args:
        metadata_list (list): List of metadata dictionaries
        
    Returns:
        dict: Metrics including averages and counts
    """
    if not metadata_list:
        return {}
    
    total_boxes = 0
    total_confidence = 0
    confidence_count = 0
    processing_times = []
    
    for metadata in metadata_list:
        # Count boxes
        boxes = metadata.get("boxes_coordinates", {})
        total_boxes += len(boxes)
        
        # Sum confidences
        for _, box_info in boxes.items():
            if isinstance(box_info, dict) and "confidence" in box_info:
                total_confidence += box_info["confidence"]
                confidence_count += 1
        
        # Track processing times
        speed_data = metadata.get("speed", {})
        total_time = sum(speed_data.values())
        processing_times.append(total_time)
    
    # Calculate averages
    avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    return {
        "total_detections": total_boxes,
        "detection_count": len(metadata_list),
        "average_confidence": avg_confidence,
        "average_processing_time": avg_processing_time,
        "min_processing_time": min(processing_times) if processing_times else 0,
        "max_processing_time": max(processing_times) if processing_times else 0
    }

# Import OpenCV here to avoid circular imports
import cv2
