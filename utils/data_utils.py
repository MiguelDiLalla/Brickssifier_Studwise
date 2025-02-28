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

# =============================================================================
# Format Conversion Functions
# =============================================================================

def convert_labelme_to_yolo(args):
    """
    Convert LabelMe JSON annotations to YOLO format for model training.
    
    Args:
        args: Command-line arguments containing:
            - input: Path to folder with LabelMe JSON files
            
    Notes:
        - Processes all JSON files in the input folder
        - Extracts bounding box coordinates and normalizes them
        - Creates corresponding YOLO format .txt files
        - Saves processed files to cache for reproducibility
    """
    input_folder = args.input
    cache_dir = "cache/datasets/processed/YOLO_labels"
    os.makedirs(cache_dir, exist_ok=True)
    output_folder = os.path.join(cache_dir, os.path.basename(input_folder))
    os.makedirs(output_folder, exist_ok=True)

    logging.info(f"Processing LabelMe annotations from: {input_folder}")
    logging.info(f"Saving YOLO annotations to: {output_folder}")

    for json_file in os.listdir(input_folder):
        if (json_file.endswith('.json')):
            json_path = os.path.join(input_folder, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logging.error(f"Failed to read {json_file}: {e}")
                continue

            image_width = data.get("imageWidth")
            image_height = data.get("imageHeight")

            if not image_width or not image_height:
                logging.warning(f"Skipping {json_file}, missing image dimensions.")
                continue

            output_path = os.path.join(output_folder, json_file.replace('.json', '.txt'))
            with open(output_path, 'w') as yolo_file:
                for shape in data.get("shapes", []):
                    points = shape.get("points", [])
                    if len(points) < 2:
                        logging.warning(f"Skipping malformed shape in {json_file}")
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
                
            logging.info(f"Converted {json_file} -> {output_path}")
    
    logging.info("LabelMe to YOLO conversion completed.")

def convert_keypoints_to_bboxes(args):
    """
    Convert keypoints from LabelMe JSON into bounding boxes.
    
    Args:
        args: Command-line arguments containing:
            - input: Path to folder with keypoint JSON files
            - area_ratio: Total area ratio for bounding boxes (default: 0.4)
            
    Notes:
        - Used for converting stud point annotations to rectangular areas
        - Calculates appropriate box size based on image dimensions and stud count
        - Preserves original image metadata while replacing annotations
    """
    input_folder = args.input
    cache_dir = "cache/datasets/processed/bboxes"
    os.makedirs(cache_dir, exist_ok=True)
    output_folder = os.path.join(cache_dir, os.path.basename(input_folder))
    os.makedirs(output_folder, exist_ok=True)
    total_area_ratio = args.area_ratio

    logging.info(f"Processing keypoints from: {input_folder}")
    logging.info(f"Saving bounding boxes to: {output_folder}")

    for json_file in os.listdir(input_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(input_folder, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logging.error(f"Failed to read {json_file}: {e}")
                continue

            image_width = data.get("imageWidth")
            image_height = data.get("imageHeight")
            image_area = image_width * image_height

            keypoints = [shape["points"][0] for shape in data.get("shapes", []) if shape["shape_type"] == "point"]
            num_keypoints = len(keypoints)

            if num_keypoints == 0:
                logging.warning(f"Skipping {json_file}, no keypoints found.")
                continue

            total_target_area = total_area_ratio * image_area
            box_area_per_keypoint = total_target_area / num_keypoints
            box_size = math.sqrt(box_area_per_keypoint)

            new_shapes = []
            for x_center, y_center in keypoints:
                x1 = max(0, x_center - box_size / 2)
                y1 = max(0, y_center - box_size / 2)
                x2 = min(image_width, x_center + box_size / 2)
                y2 = min(image_height, y_center + box_size / 2)

                new_shapes.append({
                    "label": "Stud",
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                })

            data["shapes"] = new_shapes
            output_path = os.path.join(output_folder, json_file)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            logging.info(f"Converted {json_file} -> {output_path}")
    
    logging.info("Keypoints to bounding boxes conversion completed.")

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
    labelme_parser.set_defaults(func=convert_labelme_to_yolo)

    # Convert Keypoints to Bounding Boxes
    keypoints_parser = subparsers.add_parser("keypoints-to-bboxes", help="Convert keypoints to bounding boxes.")
    keypoints_parser.add_argument("--input", required=True, help="Input folder containing keypoints JSON files.")
    keypoints_parser.add_argument("--area-ratio", type=float, default=0.4, help="Total area ratio for bounding boxes.")
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
