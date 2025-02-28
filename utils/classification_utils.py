"""
Classification Utilities for LEGO Bricks ML Vision

This module handles dimension classification of LEGO bricks
based on stud detection patterns and analysis.

Key features:
  - Dimension classification based on stud count
  - Pattern analysis to distinguish common brick shapes
  - Visualization of classification results
  - Common brick dimension mapping

Author: Miguel DiLalla
"""

import logging
import numpy as np
import cv2

# Set up logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Classification Utils module loaded.")

# Import project modules
from utils.config_utils import config

def classify_dimensions(results, orig_image, dimension_map=None):
    """
    Classifies the brick dimension based on detected stud positions.
    
    Args:
        results (list): YOLO detection results for studs
        orig_image (np.ndarray): Original image
        dimension_map (dict, optional): Mapping from stud counts to brick dimensions
        
    Returns:
        dict: Dictionary with dimension classification and annotated image
            
    Notes:
        - For simple cases, uses direct mapping from stud count to dimension
        - For complex cases with multiple possibilities:
          - Analyzes stud pattern using regression line
          - Determines if studs are in line (Nx1) or rectangular (NxM) pattern
          - Returns both dimension classification and visualization
    """
    if dimension_map is None:
        dimension_map = config["STUDS_TO_DIMENSIONS_MAP"]
    
    studs = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else np.array([])
    
    num_studs = len(studs)
    valid_stud_counts = dimension_map.keys()
    
    if num_studs == 0:
        logger.error("❌ No studs detected. Cannot classify dimensions.")
        return {
            "dimension": "Error: No studs detected",
            "annotated_image": orig_image.copy()
        }
    
    if num_studs not in valid_stud_counts:
        unknown_dimension = f"{num_studs} studs (non-standard)"
        logger.warning("⚠️ Non-standard number of studs detected (%d). Using generic label.", num_studs)
        
        # Create a basic annotated image
        annotated_image = results[0].plot(labels=False)
        cv2.putText(annotated_image, unknown_dimension, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        
        return {
            "dimension": unknown_dimension,
            "annotated_image": annotated_image
        }

    # Simple case: direct mapping available
    if num_studs in valid_stud_counts and isinstance(dimension_map[num_studs], str):
        logger.info("🧮 Detected %d studs. Dimension: %s.", num_studs, dimension_map[num_studs])
        
        # Annotate image with dimension
        annotated_image = results[0].plot(labels=False)
        h, w = annotated_image.shape[:2]
        text = f"{dimension_map[num_studs]}"
        
        # Position in bottom right with margin
        font_size = 0.8
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        text_x = w - text_size[0] - 10
        text_y = h - 10
        
        cv2.putText(annotated_image, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
        
        return {
            "dimension": dimension_map[num_studs],
            "annotated_image": annotated_image
        }
        
    # Complex case: multiple possible dimensions based on pattern
    if num_studs in valid_stud_counts and isinstance(dimension_map[num_studs], list):
        # Process pattern using analyze_stud_pattern
        pattern_result = analyze_stud_pattern(studs)
        
        # Get the appropriate dimension based on pattern type
        if pattern_result["pattern_type"] == "linear":
            # For linear patterns (Nx1), use the second option if available
            dimension = dimension_map[num_studs][1] if len(dimension_map[num_studs]) > 1 else dimension_map[num_studs][0]
        else:
            # For grid patterns (NxM), use the first option
            dimension = dimension_map[num_studs][0]
        
        logger.info("🧩 Detected %d studs in %s pattern. Dimension: %s", 
                   num_studs, pattern_result["pattern_type"], dimension)
        
        # Create annotated image
        annotated_image = orig_image.copy()
        
        # Draw centers and regression line for visualization
        centers = pattern_result["centers"]
        
        # Draw stud centers
        for i, (x, y) in enumerate(centers):
            cv2.circle(annotated_image, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(annotated_image, str(i+1), (int(x)+5, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw regression line
        if pattern_result["pattern_type"] == "linear":
            m, b = pattern_result["regression_params"]
            x1 = 0
            y1 = int(m * x1 + b)
            x2 = annotated_image.shape[1]
            y2 = int(m * x2 + b)
            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add dimension text
        h, w = annotated_image.shape[:2]
        text = f"{dimension}"
        font_size = 0.8
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        text_x = w - text_size[0] - 10
        text_y = h - 10
        cv2.putText(annotated_image, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)

        return {
            "dimension": dimension,
            "annotated_image": annotated_image,
            "pattern_analysis": pattern_result
        }
    
    # Fallback - this shouldn't happen but just in case
    logger.error("❌ Unknown case in dimension classification")
    return {
        "dimension": f"{num_studs} studs (unknown)",
        "annotated_image": results[0].plot(labels=False)
    }

def analyze_stud_pattern(studs):
    """
    Analyzes the pattern of detected studs to determine arrangement.
    
    Args:
        studs (np.ndarray): Array of bounding boxes in [x1, y1, x2, y2] format
        
    Returns:
        dict: Analysis result containing:
            - centers: List of stud center points
            - pattern_type: "linear" or "grid"
            - regression_params: (m, b) for linear regression
            - max_deviation: Maximum deviation from regression line
            
    Notes:
        - Uses linear regression to find alignment of studs
        - Determines if studs are in a line or grid pattern
        - Calculates the deviation of points from the regression line
    """
    # Extract center points from bounding boxes
    centers = []
    for stud in studs:
        x1, y1, x2, y2 = stud
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_x, center_y))
    
    # Get average stud size for threshold calculation
    avg_stud_size = np.mean([(x2-x1 + y2-y1)/2 for x1, y1, x2, y2 in studs])
    threshold = avg_stud_size / 2
    
    # Perform linear regression on centers
    if len(centers) > 1:
        xs, ys = zip(*centers)
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Calculate regression line
        m, b = np.polyfit(xs, ys, 1)
        
        # Calculate deviations from the regression line
        deviations = [abs(y - (m * x + b)) for x, y in centers]
        max_deviation = max(deviations)
        
        # Determine pattern type based on deviations
        pattern_type = "linear" if max_deviation < threshold else "grid"
    else:
        # With only one stud, we can't determine pattern
        m, b = 0, 0
        deviations = [0]
        max_deviation = 0
        pattern_type = "unknown"
    
    return {
        "centers": centers,
        "pattern_type": pattern_type,
        "regression_params": (m, b),
        "max_deviation": max_deviation,
        "deviations": deviations,
        "threshold": threshold
    }

def get_common_brick_dimensions(studs):
    """
    Gets the possible dimensions for a brick based on stud count.
    
    Args:
        studs (np.ndarray): Array of stud bounding boxes
        
    Returns:
        list: Possible brick dimension strings (e.g., ["2x2", "4x1"])
        
    Notes:
        - Returns possibilities from the dimension map
        - Falls back to a generic string for unknown stud counts
    """
    dimension_map = config["STUDS_TO_DIMENSIONS_MAP"]
    num_studs = len(studs)
    
    if num_studs in dimension_map:
        dimensions = dimension_map[num_studs]
        if isinstance(dimensions, str):
            return [dimensions]
        elif isinstance(dimensions, list):
            return dimensions
    
    # Fall back to a generic string for unknown stud counts
    return [f"{num_studs} studs"]

def estimate_brick_size(studs, pixel_to_mm=None):
    """
    Estimates the physical size of a brick based on stud detection.
    
    Args:
        studs (np.ndarray): Array of stud bounding boxes
        pixel_to_mm (float, optional): Conversion ratio from pixels to mm
        
    Returns:
        dict: Size information including width, length in pixels and mm
        
    Notes:
        - When pixel_to_mm is not provided, returns only pixel measurements
        - Standard LEGO stud spacing is 8mm
    """
    if len(studs) < 2:
        logger.warning("⚠️ Need at least 2 studs to estimate brick size.")
        return {
            "width_px": 0,
            "length_px": 0,
            "width_mm": 0,
            "length_mm": 0,
            "status": "insufficient_data"
        }
    
    # Extract center points from bounding boxes
    centers = []
    for stud in studs:
        x1, y1, x2, y2 = stud
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_x, center_y))
    
    # Get the pattern analysis to determine if it's a linear or grid pattern
    pattern_result = analyze_stud_pattern(studs)
    pattern_type = pattern_result["pattern_type"]
    
    if pattern_type == "linear":
        # For linear patterns, measure length between first and last stud
        # Sort centers by x-coordinate
        centers.sort(key=lambda p: p[0])
        first_center = centers[0]
        last_center = centers[-1]
        
        # Calculate distance
        import math
        length_px = math.sqrt((last_center[0] - first_center[0])**2 + 
                             (last_center[1] - first_center[1])**2)
        width_px = np.mean([x2 - x1 for x1, y1, x2, y2 in studs])
        
    else:  # Grid pattern
        # Find the min/max x and y coordinates
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width_px = max_x - min_x + np.mean([x2 - x1 for x1, y1, x2, y2 in studs])
        length_px = max_y - min_y + np.mean([y2 - y1 for x1, y1, x2, y2 in studs])
        
    # Convert to mm if pixel_to_mm is provided
    width_mm = width_px * pixel_to_mm if pixel_to_mm is not None else None
    length_mm = length_px * pixel_to_mm if pixel_to_mm is not None else None
    
    return {
        "width_px": width_px,
        "length_px": length_px,
        "width_mm": width_mm,
        "length_mm": length_mm,
        "pattern_type": pattern_type,
        "status": "success"
    }

def predict_dimension_from_pattern(studs, dimension_map=None):
    """
    Predicts the most likely brick dimension based on stud pattern analysis.
    
    Args:
        studs (np.ndarray): Array of stud bounding boxes
        dimension_map (dict, optional): Mapping from stud counts to dimensions
        
    Returns:
        str: Predicted dimension (e.g., "2x4")
        
    Notes:
        - Analyzes the spatial arrangement of studs
        - Uses regression to determine alignment and grid patterns
        - Considers both stud count and spatial arrangement
    """
    if dimension_map is None:
        dimension_map = config["STUDS_TO_DIMENSIONS_MAP"]
    
    num_studs = len(studs)
    
    # Direct mapping case - one possible dimension
    if num_studs in dimension_map and isinstance(dimension_map[num_studs], str):
        return dimension_map[num_studs]
    
    # No dimension mapping available
    if num_studs not in dimension_map:
        return f"{num_studs} studs (non-standard)"
    
    # Multiple possible dimensions - analyze pattern
    pattern_result = analyze_stud_pattern(studs)
    pattern_type = pattern_result["pattern_type"]
    
    # Get dimensions from map
    possible_dimensions = dimension_map[num_studs]
    
    if pattern_type == "linear":
        # For linear patterns, prefer the Nx1 format (e.g., "4x1" over "2x2")
        for dim in possible_dimensions:
            if "x1" in dim:
                return dim
        # If no Nx1 format found, return first option
        return possible_dimensions[0]
    else:
        # For grid patterns, prefer the more square format (e.g., "2x2" over "4x1")
        for dim in possible_dimensions:
            if "x1" not in dim:
                return dim
        # If no grid format found, return first option
        return possible_dimensions[0]
