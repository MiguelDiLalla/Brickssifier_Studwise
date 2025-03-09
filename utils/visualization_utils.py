"""
Visualization Utilities for LEGO Bricks ML Vision

This module handles image annotation, visualization, and rendering
for the LEGO Bricks ML Vision project.

Key features:
  - Annotating images with detection results
  - Creating composite images
  - Visualizing dimension classifications

Author: Miguel DiLalla
"""

import logging
import cv2
import numpy as np

# Set up logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Visualization Utils module loaded.")

# Import project modules
from utils.config_utils import config
from utils.exif_utils import read_exif
from rich.table import Table
from rich.console import Console

def annotate_scanned_image(image_path):
    """
    Annotates the provided image with the metadata retrieved via read_exif().
    
    Args:
        image_path (str): Path to the image file to annotate
        
    Returns:
        np.ndarray: The annotated image with bounding boxes and project logo
        
    Notes:
        - Reads metadata directly from image EXIF data
        - Applies colored bounding boxes for detections
        - Adds the project logo to the bottom of the image
    """
    # Load image from path
    image = cv2.imread(image_path)
    if image is None:
        logger.error("❌ Failed to load image for annotation: %s", image_path)
        return None

    # Retrieve full metadata using read_exif()
    metadata = read_exif(image_path)

    label = metadata.get('mode', 'Label not found')
    if label != 'Label not found':
        label = label[:-1]

    for key, box_info in metadata.get('boxes_coordinates', {}).items():
        if isinstance(box_info, dict):
            coords = box_info.get("coordinates", [])
        elif isinstance(box_info, list):
            coords = box_info
        else:
            coords = []

        if len(coords) == 4:
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Add the logo to the bottom of the image
    logo = config["LOGO_NUMPY"]
    if logo is not None:
        # Convert the logo from black and white to red and white:
        colored_logo = logo.copy()
        # Identify "black" pixels (assumes near-zero values indicate black)
        black_mask = (colored_logo[:, :, 0] < 128) & (colored_logo[:, :, 1] < 128) & (colored_logo[:, :, 2] < 128)
        # Set these black pixels to red (BGR: (0, 0, 255))
        colored_logo[black_mask] = [0, 0, 255]
        logo = colored_logo

        img_h, img_w, _ = image.shape
        logo_height, logo_width, _ = logo.shape
        # Resize the annotated image to match the logo's width (preserve aspect ratio)
        new_img_h = int(img_h * (logo_width / img_w))
        resized_image = cv2.resize(image, (logo_width, new_img_h))
        # Combine the resized image and the logo vertically
        composite_image = np.vstack((resized_image, logo))
        # Add a red margin to the composite image
        composite_image = cv2.copyMakeBorder(composite_image, 10, 10, 10, 10,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 255))
        
        # print all paths avalaible in the metadata after validating each one
        for key, value in metadata.items():
            if key == "annotated_image_path" and value != "":
                logger.info("📂 Annotated image path: %s", value)
            if key == "json_results_path" and value != "":
                logger.info("📂 JSON results path: %s", value)
            if key == "path" and value != "":
                logger.info("📂 Original image path: %s", value)

        return composite_image
    
    else:
        logger.warning("⚠️ No logo available for annotation.")

    return image

def read_detection(image_path):
    """
    Reads the detection results from the image's EXIF metadata.
    
    Args:
        image_path (str): Path to the image file to read
        
    Returns:
        dict: Dictionary containing:
            - orig_image: Original image
            - annotated_image: Image with annotations
            - cropped_detections: List of cropped detections
            - metadata: Complete metadata dictionary
            - status: Processing status
            - dimensions: (Only for "stud" detection) Brick dimensions
    """
    metadata = read_exif(image_path)
    orig_image = cv2.imread(image_path)
    
    if not orig_image is not None:
        logger.error("❌ Failed to load image from path: %s", image_path)
        return None
    
    # Create a copy for annotation
    annotated_image = orig_image.copy()
    cropped_detections = []

    # Check if metadata exists and has detection information
    if not metadata:
        logger.warning("⚠️ No EXIF metadata found in the image: %s", image_path)
        return {
            "orig_image": orig_image,
            "annotated_image": orig_image,  # No annotations since no detections
            "cropped_detections": [],
            "metadata": {},
            "status": "no_metadata"
        }

    if "boxes_coordinates" not in metadata or not metadata["boxes_coordinates"]:
        logger.warning("⚠️ No detection information found in metadata for: %s", image_path)
        return {
            "orig_image": orig_image,
            "annotated_image": orig_image,  # No annotations since no detections
            "cropped_detections": [],
            "metadata": metadata,
            "status": "no_detections"
        }
    
    name = metadata.get("mode", "Label not found")
    label = name[:-1] if name == 'bricks' else name

    # Process valid metadata with detections
    for key, box_info in metadata["boxes_coordinates"].items():
        if isinstance(box_info, dict):
            coords = box_info.get("coordinates", [])
        elif isinstance(box_info, list):
            coords = box_info
        else:
            coords = []

        if len(coords) == 4:
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            conf = f"{box_info.get('confidence', 0.0):.2f}"
            # Reduce font size by half if label is "stud"
            font_size = 0.25 if label == "stud" else 0.5
            cv2.putText(annotated_image, f"{label} {conf}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
            
            crop = orig_image[y1:y2, x1:x2]
            cropped_detections.append(crop) 
            
            logger.info(f"📦 Detected crop from coordinates: {coords}")
        else:
            logger.warning("⚠️ Invalid coordinates found in metadata: %s", coords)
        
        # at the bottom right of the image add "NO MODEL RAN"
        text_position = (annotated_image.shape[1] - 150, annotated_image.shape[0] - 10)
        cv2.putText(annotated_image, "NO MODEL RAN", text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # TYPE the dimensions detected if the label is "stud"
    if label == "stud":
        dimension = metadata.get("dimension", "Dimensions not found")
        # Position the dimension text at the top left of the image
        text_position = (10, 20)
        cv2.putText(annotated_image, f"{dimension}", text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        logger.info(f"📏 Dimensions for 'stud': {dimension}")
        return {
            "orig_image": orig_image,
            "annotated_image": annotated_image,
            # "cropped_detections": cropped_detections,
            "metadata": metadata,
            "dimensions": dimension,
            "status": "success"
        }
    else:
        return {
            "orig_image": orig_image,
            "annotated_image": annotated_image,
            "cropped_detections": cropped_detections,
            "metadata": metadata,
            "status": "success"
        }

def create_composite_image(base_image, detection_images, logo=None):
    """
    Creates a composite image showing the base image with all detections.

    Args:
        base_image (np.ndarray): The original image with brick detections
        detection_images (list): List of images showing stud detections
        logo (np.ndarray, optional): Logo to add at the bottom of the image

    Returns:
        np.ndarray: Composite image with all visualizations
    """
    if not detection_images:
        logger.warning("⚠️ No detection images provided for composite.")
        composite_image = base_image.copy()
    elif len(detection_images) == 1:
        # For a single detection image
        studs_image = detection_images[0]
        
        # Resize studs image to match base image width
        h, w = studs_image.shape[:2]
        new_w = base_image.shape[1]
        new_h = int(h * new_w / w)
        resized_studs_image = cv2.resize(studs_image, (new_w, new_h))
        
        # Stack base image and studs image vertically
        composite_image = np.vstack((base_image, resized_studs_image))
    else:
        # For multiple detection images, sort by size and stack horizontally
        areas = [image.shape[0] * image.shape[1] for image in detection_images]
        sorted_images = [image for _, image in sorted(zip(areas, detection_images), reverse=True)]
        
        # Resize all images to the same height
        height = sorted_images[0].shape[0]
        resized_images = [cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height)) 
                         for image in sorted_images]
        
        # Stack images horizontally
        studs_image = np.hstack(resized_images)
        
        # Resize studs image to match base image width
        width_ratio = base_image.shape[1] / studs_image.shape[1]
        new_width = base_image.shape[1]
        new_height = int(studs_image.shape[0] * width_ratio)
        studs_image = cv2.resize(studs_image, (new_width, new_height))
        
        # Stack base image and studs image vertically
        composite_image = np.vstack((base_image, studs_image))

    # Add logo to the bottom if provided
    if logo is not None:
        # Resize logo to match the width of the composite image
        logo_height, logo_width = logo.shape[:2]
        composite_width = composite_image.shape[1]
        aspect_ratio = logo_width / logo_height
        new_logo_height = int(composite_width / aspect_ratio)
        resized_logo = cv2.resize(logo, (composite_width, new_logo_height))
        
        # Convert to float for blending calculation
        logo_float = resized_logo.astype(float)
        
        # Create a red background
        background = np.zeros_like(logo_float)
        background[:,:,2] = 255.0  # Red in BGR
        
        # Screen blend formula
        blended_logo = 255 - ((255 - background) * (255 - logo_float) / 255)
        blended_logo = blended_logo.astype(np.uint8)
        
        # Stack the blended logo below the composite image
        composite_image = np.vstack((composite_image, blended_logo))
    else:
        logger.warning("⚠️ No logo available for the composite image.")

    # Add a red margin to the composite image
    composite_image = cv2.copyMakeBorder(composite_image, 10, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 255))
    
    return composite_image

def draw_detection_visualization(image, boxes, class_names=None, confidence_scores=None, color=(0, 255, 0)):
    """
    Draws bounding boxes and labels on an image.
    
    Args:
        image (np.ndarray): Image to annotate
        boxes (np.ndarray): Bounding boxes in [x1, y1, x2, y2] format
        class_names (list, optional): Class names for each box
        confidence_scores (list, optional): Confidence scores for each box
        color (tuple): BGR color for bounding boxes
        
    Returns:
        np.ndarray: Annotated image
    """
    annotated = image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        label = ""
        if class_names is not None and i < len(class_names):
            label += class_names[i]
        
        if confidence_scores is not None and i < len(confidence_scores):
            if label:
                label += " "
            label += f"{confidence_scores[i]:.2f}"
        
        if label:
            # Position label at the top of the box
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_offset_x = x1
            text_offset_y = y1 - 5
            
            # Draw background rectangle for text
            cv2.rectangle(annotated, 
                         (text_offset_x, text_offset_y - text_height),
                         (text_offset_x + text_width, text_offset_y + 5),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated, label, 
                       (text_offset_x, text_offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 0, 0), thickness)
    
    return annotated

def display_results_table(brick_results, studs_results):
    """
    Displays a formatted table of detection results using Rich.
    
    Args:
        brick_results (dict): Results from brick detection containing cropped_detections
        studs_results (list): List of results from stud detection per brick
    """
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Detection Type", style="dim")
    table.add_column("Count")
    table.add_column("Details")
    
    # Add brick detection results
    brick_count = len(brick_results.get("cropped_detections", []))
    table.add_row(
        "Bricks",
        str(brick_count),
        "Confidence scores available in brick_results"
    )
    
    # Add stud detection results
    total_studs = sum(len(result.get("boxes", [])) for result in studs_results)
    stud_details = [
        f"Brick {i+1}: {len(result.get('boxes', []))} studs" 
        for i, result in enumerate(studs_results)
    ]
    table.add_row(
        "Studs",
        str(total_studs),
        "\n".join(stud_details)
    )
    
    console.print(table)