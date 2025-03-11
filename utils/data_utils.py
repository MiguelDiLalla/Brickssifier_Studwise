"""
Data Utilities for LEGO Bricks ML Vision

This module provides dataset processing utilities for preparing, converting,
and visualizing annotated image data for the LEGO Bricks ML Vision project.

Key features:
  - Conversion between annotation formats (LabelMe to YOLO)
  - Keypoint conversion to bounding boxes
  - Dataset visualization tools
  - Structured logging with error handling

Author: Miguel DiLalla
"""

import os
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import random
import base64
import numpy as np
import cv2
import random
from datetime import datetime

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging():
    """
    Configures structured logging for data processing operations.
    
    Sets up a StreamHandler with formatted output for clear progress tracking.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

# Initialize logger
logger = logging.getLogger(__name__)
setup_logging()

# =============================================================================
# Format Conversion Functions
# =============================================================================

def _convert_keypoint_to_bbox(point, image_width, image_height, box_size_ratio=0.02, total_points=1):
    """Helper function to convert a keypoint to bbox coordinates.
    
    Args:
        point (list): [x, y] coordinates of the keypoint
        image_width (int): Width of the image
        image_height (int): Height of the image
        box_size_ratio (float): Total target area ratio for all boxes combined
        total_points (int): Total number of points to distribute area between
        
    Returns:
        tuple: (x_center, y_center, width, height) normalized coordinates
    """
    x, y = point
    # Calculate box size using the same logic as convert_keypoints_to_bboxes
    total_target_area = box_size_ratio * image_width * image_height
    box_area = total_target_area / total_points
    box_size = math.sqrt(box_area)
    
    # Ensure box stays within image bounds
    x1 = max(0, x - box_size/2)
    y1 = max(0, y - box_size/2)
    x2 = min(image_width, x + box_size/2)
    y2 = min(image_height, y + box_size/2)
    
    # Convert to normalized coordinates
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    
    return x_center, y_center, width, height

def convert_labelme_to_yolo(args):
    """
    Convert LabelMe JSON annotations to YOLO format.
    
    Args:
        args: Command arguments containing:
            input: Path to input folder
            output: Path to output folder
            clean: Whether to clean output directory first
            
    Returns:
        dict: Conversion statistics including:
            total: Total files processed
            success: Successfully converted files
            failed: Failed conversions
            output_path: Path to output directory
    """
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'output_path': args.output
    }
    
    os.makedirs(args.output, exist_ok=True)
    if args.clean:
        for f in os.listdir(args.output):
            os.remove(os.path.join(args.output, f))
    
    for json_file in os.listdir(args.input):
        if not json_file.endswith('.json'):
            continue
            
        stats['total'] += 1
        json_path = os.path.join(args.input, json_file)
        output_path = os.path.join(args.output, json_file.replace('.json', '.txt'))
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            image_width = data.get("imageWidth")
            image_height = data.get("imageHeight")
            
            if not image_width or not image_height:
                raise ValueError("Missing image dimensions")
                
            with open(output_path, 'w') as yolo_file:
                for shape in data.get("shapes", []):
                    points = shape.get("points", [])
                    if len(points) < 2:
                        continue
                        
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x_min, x_max = sorted([x1, x2])
                    y_min, y_max = sorted([y1, y2])
                    
                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height
                    
                    yolo_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            stats['success'] += 1
            
        except Exception as e:
            logging.error(f"Failed to convert {json_file}: {str(e)}")
            stats['failed'] += 1
            if os.path.exists(output_path):
                os.remove(output_path)
    
    return stats

def convert_keypoints_to_bboxes(args):
    """
    Convert keypoint annotations to bounding boxes.
    
    Args:
        args: Command arguments containing:
            input: Path to input folder
            output: Path to output folder
            area_ratio: Target area ratio for boxes
            clean: Whether to clean output directory first
            
    Returns:
        dict: Conversion statistics
    """
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'output_path': args.output
    }
    
    os.makedirs(args.output, exist_ok=True)
    if args.clean:
        for f in os.listdir(args.output):
            os.remove(os.path.join(args.output, f))
    
    for json_file in os.listdir(args.input):
        if not json_file.endswith('.json'):
            continue
            
        stats['total'] += 1
        try:
            with open(os.path.join(args.input, json_file), 'r') as f:
                data = json.load(f)
            
            image_width = data.get("imageWidth")
            image_height = data.get("imageHeight")
            if not image_width or not image_height:
                raise ValueError("Missing image dimensions")
            
            keypoints = [
                shape["points"][0] 
                for shape in data.get("shapes", [])
                if shape["shape_type"] == "point"
            ]
            
            total_points = len(keypoints)
            new_shapes = []
            for point in keypoints:
                # Use helper function with consistent area ratio logic
                x_center, y_center, width, height = _convert_keypoint_to_bbox(
                    point,
                    image_width,
                    image_height,
                    box_size_ratio=args.area_ratio,
                    total_points=total_points
                )
                
                # Convert normalized coordinates back to absolute
                x1 = int((x_center - width/2) * image_width)
                y1 = int((y_center - height/2) * image_height)
                x2 = int((x_center + width/2) * image_width)
                y2 = int((y_center + width/2) * image_height)
                
                new_shapes.append({
                    "label": "Stud",
                    "points": [[x1, y1], [x2, y2]],
                    "shape_type": "rectangle",
                    "flags": {}
                })
            
            data["shapes"] = new_shapes
            output_path = os.path.join(args.output, json_file)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            stats['success'] += 1
            
        except Exception as e:
            logging.error(f"Failed to convert {json_file}: {str(e)}")
            stats['failed'] += 1
            
    return stats

def convert_labelme_to_yolo_format(json_data):
    """Convert LabelMe annotations to YOLO format strings.
    
    Args:
        json_data (dict): LabelMe JSON data
        
    Returns:
        list: YOLO format annotation strings
    """
    image_width = json_data["imageWidth"]
    image_height = json_data["imageHeight"]
    yolo_labels = []
    
    for shape in json_data["shapes"]:
        if shape["shape_type"] == "rectangle":
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        elif shape["shape_type"] == "point":
            # Convert keypoint to bbox using helper function
            x_center, y_center, width, height = _convert_keypoint_to_bbox(
                shape["points"][0],
                image_width,
                image_height
            )
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_labels

def _validate_base64_image(base64_str):
    """Validate base64 image data.
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not base64_str:
        return False
        
    # Check if string has valid base64 pattern
    try:
        # Remove common base64 image prefixes if present
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        # Try decoding a small sample to validate format
        base64.b64decode(base64_str[:64])
        return True
    except Exception:
        return False

def _generate_procedural_pattern(width, height):
    """Generate a procedural Turing pattern for placeholder images.
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        ndarray: Generated RGB image with Turing pattern
    """
    # Generate random saturated and dark colors
    # Generate bright color using HSV
    hue = random.randint(0, 360)  # Full hue range
    saturation = random.randint(80, 100) / 100  # High saturation
    brightness = random.randint(90, 100) / 100  # High brightness
    
    # Convert HSV to RGB (OpenCV uses BGR)
    hsv = np.array([[[hue/2, saturation*255, brightness*255]]], dtype=np.uint8)  # OpenCV uses H=0-180
    bright_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    
    # Generate dark color using HSV - different hue, lower brightness
    dark_hue = (hue + 180) % 360  # Opposite hue on the color wheel
    dark_hsv = np.array([[[dark_hue/2, saturation*255, brightness*255*0.3]]], dtype=np.uint8)  # 30% brightness
    dark_color = cv2.cvtColor(dark_hsv, cv2.COLOR_HSV2BGR)[0][0]
    
    # Create base noise pattern
    pattern = np.random.rand(height, width)
    
    # Apply simple diffusion to create Turing-like pattern
    kernel = np.ones((3,3)) / 9.0
    for _ in range(3):  # Apply diffusion multiple times
        pattern = cv2.filter2D(pattern, -1, kernel)
    
    # Threshold pattern to create binary image
    pattern = (pattern > 0.5).astype(np.uint8)
    
    # Create RGB image from pattern
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[pattern == 1] = bright_color
    result[pattern == 0] = dark_color
    
    return result

def create_conversion_demo(json_path, output_folder):
    """Create a visual demo of the annotation conversion process.
    
    Shows a side-by-side comparison of:
    - Left: Grayscale image with original LabelMe annotations
    - Right: Color image with converted YOLO annotations
    
    Args:
        json_path (str): Path to LabelMe JSON file
        output_folder (str): Output directory for demo files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input file
        if not os.path.exists(json_path):
            logger.error(f"Input file not found: {json_path}")
            return False
            
        # Load JSON data
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {json_path}: {str(e)}")
            return False
            
        # Validate required fields
        if 'imageData' not in json_data:
            logger.error(f"Missing 'imageData' field in {json_path}")
            return False
            
        if not _validate_base64_image(json_data['imageData']):
            logger.error(f"Invalid base64 image data in {json_path}")
            image = _generate_procedural_pattern(json_data["imageWidth"], json_data["imageHeight"])
            
        # Decode base64 image with explicit error handling
        if json_data['imageData'] is not None:
            if not locals().get('image'):
                try:
                    img_data = base64.b64decode(json_data['imageData'])
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    # #log imageData key
                    # logger.info(f"imageData key found in {json_data['imageData']}")
                    
                    if image is None:
                        raise ValueError("Failed to decode image data")
                except Exception as e:
                    logger.warning(f"Failed to decode image from base64, using procedural pattern: {str(e)}")
                    image = _generate_procedural_pattern(json_data["imageWidth"], json_data["imageHeight"])
        else:
            logger.info("No image data found, using procedural pattern")
            image = _generate_procedural_pattern(json_data["imageWidth"], json_data["imageHeight"])
            
        # Validate image dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            logger.error(f"Invalid image dimensions in {json_path}")
            return False
            
        # Prepare images for visualization
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        yolo_viz = image.copy()
        
        # Draw LabelMe annotations on grayscale image
        labelme_viz = gray_rgb.copy()
        for shape in json_data['shapes']:
            points = np.array(shape['points'], dtype=np.int32)
            label = shape['label']
            
            if shape['shape_type'] == 'rectangle':
                pt1 = tuple(points[0])
                pt2 = tuple(points[1])
                cv2.rectangle(labelme_viz, pt1, pt2, (0, 255, 0), 2)
                # Add original label above the box
                label_pos = (pt1[0], pt1[1] - 5)
                cv2.putText(labelme_viz, label, label_pos, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)
                
            elif shape['shape_type'] == 'point':
                center = tuple(points[0])
                cv2.circle(labelme_viz, center, 3, (0, 255, 0), -1)
                # Add label next to point
                label_pos = (center[0] + 5, center[1] + 5)
                cv2.putText(labelme_viz, label, label_pos, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to and draw YOLO format annotations
        yolo_labels = convert_labelme_to_yolo_format(json_data)
        img_height, img_width = image.shape[:2]
        
        for label in yolo_labels:
            class_id, x_center, y_center, width, height = map(float, label.split())
            
            # Convert normalized coordinates back to pixel coordinates
            x = int((x_center - width/2) * img_width)
            y = int((y_center - height/2) * img_height)
            w = int(width * img_width)
            h = int(height * img_height)
            
            # Draw bounding box 
            cv2.rectangle(yolo_viz, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Add actual class ID above the box
            label_pos = (x, y - 5)
            cv2.putText(yolo_viz, str(int(class_id)), label_pos, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
        # Create side-by-side comparison
        demo_image = np.hstack([labelme_viz, yolo_viz])
        
        # Add labels at the top
        font = cv2.FONT_HERSHEY_TRIPLEX
        y_pos = 30
        cv2.putText(demo_image, 'LabelMe Annotations', (10, y_pos), font, 1, (255, 255, 255), 2)
        cv2.putText(demo_image, 'YOLO Format', (img_width + 10, y_pos), font, 1, (255, 255, 255), 2)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Save comparison image
        output_path = os.path.join(output_folder, f"{base_name}_comparison.jpg")
        labelme_txt = os.path.join(output_folder, f"{base_name}_labelme.txt")
        yolo_txt = os.path.join(output_folder, f"{base_name}_yolo.txt")
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Save files with error handling
        try:
            cv2.imwrite(output_path, demo_image)
            
            with open(labelme_txt, 'w') as f:
                json.dump(json_data['shapes'], f, indent=2)
                
            with open(yolo_txt, 'w') as f:
                f.write('\n'.join(yolo_labels))
                
        except Exception as e:
            logger.error(f"Failed to save output files: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error processing {json_path}: {str(e)}")
        return False

# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_yolo_annotation(args):
    """
    Displays YOLO annotations overlaid on corresponding images.
    
    Args:
        args: Command-line arguments containing:
            - input: Path to image file or folder
            - labels: Path to folder containing YOLO .txt labels
            - grid_size: Visualization grid dimensions (e.g., "3x4")
            
    Notes:
        - Creates a grid display of randomly selected images (if folder provided)
        - Draws bounding boxes based on YOLO coordinates
        - Useful for validating dataset annotations before training
    """
    image_path_or_folder = args.input
    labels_folder = args.labels
    grid_size = tuple(map(int, args.grid_size.split('x')))

    if os.path.isdir(image_path_or_folder):
        image_files = [f for f in os.listdir(image_path_or_folder) if f.endswith(('.jpg', '.png'))]
        if not image_files:
            logging.error("No images found in the folder.")
            return
        random.shuffle(image_files)
        selected_images = image_files[:grid_size[0] * grid_size[1]]
    else:
        selected_images = [os.path.basename(image_path_or_folder)]
        image_path_or_folder = os.path.dirname(image_path_or_folder)
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 8))
    axes = axes.flatten()
    
    for ax, img_name in zip(axes, selected_images):
        img_path = os.path.join(image_path_or_folder, img_name)
        label_path = os.path.join(labels_folder, img_name.replace(os.path.splitext(img_name)[-1], ".txt"))
        
        if not os.path.exists(label_path):
            logging.warning(f"Skipping {img_name}: No matching label file found.")
            continue
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                _, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                x1 = int((x_center - bbox_width / 2) * width)
                y1 = int((y_center - bbox_height / 2) * height)
                x2 = int((x_center + bbox_width / 2) * width)
                y2 = int((y_center + bbox_height / 2) * height)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(img_name)
    
    plt.tight_layout()
    plt.show()
    logging.info("YOLO annotations visualization completed.")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Command-line interface for data processing utilities.
    
    Provides subcommands for:
    - labelme-to-yolo: Convert LabelMe JSON to YOLO format
    - keypoints-to-bboxes: Convert keypoint annotations to bounding boxes
    - visualize: Display annotated images with bounding boxes
    """
    parser = argparse.ArgumentParser(description="Data Utilities for LEGO ML Project")
    subparsers = parser.add_subparsers(dest="command")

    # Convert LabelMe to YOLO
    labelme_parser = subparsers.add_parser("labelme-to-yolo", help="Convert LabelMe JSON annotations to YOLO format.")
    labelme_parser.add_argument("--input", required=True, help="Input folder containing LabelMe JSON files.")
    labelme_parser.add_argument("--output", required=True, help="Output folder for YOLO .txt files.")
    labelme_parser.add_argument("--clean", action="store_true", help="Clean output directory before conversion.")
    labelme_parser.set_defaults(func=convert_labelme_to_yolo)

    # Convert Keypoints to Bounding Boxes
    keypoints_parser = subparsers.add_parser("keypoints-to-bboxes", help="Convert keypoints to bounding boxes.")
    keypoints_parser.add_argument("--input", required=True, help="Input folder containing keypoints JSON files.")
    keypoints_parser.add_argument("--output", required=True, help="Output folder for bounding box JSON files.")
    keypoints_parser.add_argument("--area-ratio", type=float, default=0.4, help="Total area ratio for bounding boxes.")
    keypoints_parser.add_argument("--clean", action="store_true", help="Clean output directory before conversion.")
    keypoints_parser.set_defaults(func=convert_keypoints_to_bboxes)

    # Visualize YOLO Annotations
    visualize_parser = subparsers.add_parser("visualize", help="Visualize YOLO annotations.")
    visualize_parser.add_argument("--input", required=True, help="Path to a single image or folder of images.")
    visualize_parser.add_argument("--labels", required=True, help="Folder containing YOLO .txt labels.")
    visualize_parser.add_argument("--grid-size", default="3x4", help="Grid size for visualization (e.g., 3x4).")
    visualize_parser.set_defaults(func=visualize_yolo_annotation)

    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
