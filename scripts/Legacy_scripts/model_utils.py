"""
Model Utilities for LEGO Bricks ML Vision

This module provides core machine learning utilities for the LEGO Bricks ML Vision project.
It contains functions for model loading, inference, metadata handling, and result visualization.

Key features:
  - YOLO model loading and inference for brick and stud detection
  - EXIF metadata reading/writing with scan count tracking
  - Image annotation and visualization capabilities
  - Dimension classification based on stud detection
  - Rich logging with emoji markers for readability

Usage examples:
  - detect_bricks() - Detect LEGO bricks in images
  - detect_studs() - Detect studs on LEGO bricks
  - run_full_algorithm() - Run the complete detection pipeline

Author: Miguel DiLalla
"""

import json
import logging
import datetime
import platform
import os
import base64
import sys
import requests
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any

import numpy as np
import torch
import cv2
import piexif
from PIL import Image, ImageDraw, ImageFont, ExifTags
from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, track, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.status import Status
from rich.text import Text
from rich.style import Style
from rich.layout import Layout




# Import project modules - will gracefully handle if rich_utils is not available
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
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        RICH_AVAILABLE = True
        console = Console()
    except ImportError:
        RICH_AVAILABLE = False
        print("Warning: 'rich' package not available. Install with: pip install rich")

# Set up professional logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Model Utils module loaded.")

# =============================================================================
# Configuration and Setup
# =============================================================================

def setup_utils(repo_download=False):
    """
    Initialize and return a configuration dictionary with all global variables and defaults.
    
    Args:
        repo_download (bool): Whether to download missing assets from repository.
        
    Returns:
        dict: Configuration dictionary with paths, models, and settings.
    """
    CONFIG_DICT = {}  # New configuration dictionary
    
    # Project repository configuration
    userGithub = "MiguelDiLalla"
    repoGithub = "LEGO_Bricks_ML_Vision"
    CONFIG_DICT["REPO_URL"] = f"https://api.github.com/repos/{userGithub}/{repoGithub}/contents/"
    logger.info("📌 REPO URL set to: %s", CONFIG_DICT["REPO_URL"])
    
    # Define model and test images folders relative to project structure.
    CONFIG_DICT["MODELS_PATHS"] = {
        "bricks": r"presentation/Models_DEMO/Brick_Model_best20250123_192838t.pt",
        "studs": r"presentation/Models_DEMO/Stud_Model_best20250124_170824.pt"
    }
    CONFIG_DICT["TEST_IMAGES_FOLDERS"] = {
        "bricks": r"presentation/Test_images/BricksPics",
        "studs": r"presentation/Test_images/StudsPics"
    }
    
    logger.info("📂 Current working directory: %s", os.getcwd())

    def get_image_files(folder):
        """Get all image files from a folder with supported extensions."""
        full_path = os.path.join(os.getcwd(), folder)
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        if not os.path.exists(full_path):
            return []
        return [os.path.join(full_path, f) for f in os.listdir(full_path) if f.lower().endswith(image_extensions)]
    
    CONFIG_DICT["TEST_IMAGES"] = {}
    for key, folder in CONFIG_DICT["TEST_IMAGES_FOLDERS"].items():
        full_folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(full_folder_path):
            files = get_image_files(folder)
            CONFIG_DICT["TEST_IMAGES"][key] = files
            logger.info("✅ Found %d images in %s", len(files), folder)
        else:
            CONFIG_DICT["TEST_IMAGES"][key] = []
            logger.warning("⚠️ Folder %s does not exist; no images found.", folder)
    
    # Load models from disk or download them if needed
    CONFIG_DICT["LOADED_MODELS"] = {}
    for model_name, relative_path in CONFIG_DICT["MODELS_PATHS"].items():
        local_path = os.path.join(os.getcwd(), relative_path)
        if not os.path.exists(local_path):
            if repo_download:
                model_url = CONFIG_DICT["REPO_URL"] + relative_path
                logger.info("⬇️  Downloading %s model from %s", model_name, model_url)
                response = requests.get(model_url)
                if response.status_code == 200:
                    data = response.json()
                    model_data = base64.b64decode(data.get("content", ""))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as model_file:
                        model_file.write(model_data)
                else:
                    logger.error("❌ Failed to download %s model from %s", model_name, model_url)
            else:
                logger.error("❌ %s model not found locally and repo_download is disabled.", model_name)
        try:
            CONFIG_DICT["LOADED_MODELS"][model_name] = YOLO(local_path)
            logger.info("✅ %s model loaded.", model_name.capitalize())
        except Exception as e:
            logger.error("❌ Error loading %s model: %s", model_name, e)
            CONFIG_DICT["LOADED_MODELS"][model_name] = None
    
    # Retrieve project logo
    try:
        local_logo_path = os.path.join(os.getcwd(), "presentation", "logo.png")
        if os.path.exists(local_logo_path):
            CONFIG_DICT["LOGO_NUMPY"] = cv2.imread(local_logo_path)
            logger.info("🖼️ Logo found locally.")
        else:
            if repo_download:
                logo_url = CONFIG_DICT["REPO_URL"] + "presentation/logo.png"
                logger.info("⬇️ Logo not found locally. Downloading from %s", logo_url)
                response = requests.get(logo_url)
                if response.status_code == 200:
                    data = response.json()
                    logo_data = base64.b64decode(data.get("content", ""))
                    os.makedirs(os.path.dirname(local_logo_path), exist_ok=True)
                    with open(local_logo_path, "wb") as logo_file:
                        logo_file.write(logo_data)
                    CONFIG_DICT["LOGO_NUMPY"] = cv2.imread(local_logo_path)
                else:
                    logger.error("❌ Failed to download logo from %s", logo_url)
                    CONFIG_DICT["LOGO_NUMPY"] = None
            else:
                logger.error("❌ Logo not found locally and repo_download is disabled.")
                CONFIG_DICT["LOGO_NUMPY"] = None
    except Exception as e:
        logger.error("❌ Error loading logo: %s", e)
        CONFIG_DICT["LOGO_NUMPY"] = None

    # Mapping for studs to brick dimensions.
    CONFIG_DICT["STUDS_TO_DIMENSIONS_MAP"] = {
        1: "1x1",
        2: "2x1",
        3: "3x1",
        4: ["2x2", "4x1"],
        6: ["3x2", "6x1"],
        8: ["4x2", "8x1"],
        10: "10x1",
        12: ["6x2", "12x1"],
        16: ["4x4", "8x2"]
    }

    # Default EXIF metadata backbone.
    CONFIG_DICT["EXIF_METADATA_DEFINITIONS"] = {
        "boxes_coordinates": {},       # Detected bounding box coordinates.
        "orig_shape": [0, 0],            # Original image dimensions.
        "speed": {                     # Processing time metrics.
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0
        },
        "mode": "",                    # Operation mode: detection/classification.
        "path": "",                    # Original image file path.
        "os_full_version_name": "",    # OS version information.
        "processor": "",               # Processor details.
        "architecture": "",            # System architecture.
        "hostname": "",                # Host machine name.
        "timestamp": "",               # Time of processing.
        "annotated_image_path": "",    # Path for annotated output.
        "json_results_path": "",       # Path for exported metadata.
        "TimesScanned": 0,             # Number of inference sessions.
        "Repository": CONFIG_DICT["REPO_URL"],        # Repository URL.
        "message": ""                  # Custom message.
    }
    
    return CONFIG_DICT

config = setup_utils()

# =============================================================================
# EXIF Functions
# =============================================================================

def read_exif(image_path, TREE=config["EXIF_METADATA_DEFINITIONS"]):
    """
    Reads EXIF metadata from an image file and logs scan status.
    
    Args:
        image_path (str): Path to the image file to read metadata from
        TREE (dict, optional): Dictionary structure for default values if metadata is missing
            Defaults to EXIF_METADATA_DEFINITIONS from config.
            
    Returns:
        dict: Parsed metadata or empty dictionary if no metadata found
    
    Notes:
        - Handles special case for "image0.jpg" as numpy array source
        - Updates TimesScanned for tracking processing history
    """
    # Use default EXIF DEFINITIONS from CONFIG if TREE is None
    if "image0.jpg" in image_path :
        logger.info("🆕 Using numpy array source.")
        return {}

    if TREE is None:
        TREE = config["EXIF_METADATA_DEFINITIONS"]

    try:
        with Image.open(image_path) as image:
            exif_bytes = image.info.get("exif")
            if not exif_bytes:
                logger.warning("⚠️ No EXIF data found in %s", image_path)
                return {}

            exif_dict = piexif.load(exif_bytes)
    except Exception as e:
        logger.error("❌ Failed to open image %s > %s", image_path, e)
        return {}

    user_comment_tag = piexif.ExifIFD.UserComment
    user_comment = exif_dict.get("Exif", {}).get(user_comment_tag, b"")
    if not user_comment:
        logger.warning("⚠️ No UserComment tag found in %s", image_path)
        return {}

    try:
        comment_str = user_comment.decode('utf-8', errors='ignore')
        metadata = json.loads(comment_str)
        # Ensure defaults from TREE are present
        for key, default in TREE.items():
            metadata.setdefault(key, default)
        times = metadata.get("TimesScanned", 0)
        if times:
            logger.info("🔄 Image %s has been scanned %d time(s)", image_path, times)
        else:
            logger.info("🆕 Image %s has not been scanned before", image_path)
        return metadata
    except Exception as e:
        logger.error("❌ Failed to parse EXIF metadata from %s: %s", image_path, e)
        return {}
    
def write_exif(image_path, metadata):
    """
    Writes metadata to the image's EXIF UserComment tag.
    
    Args:
        image_path (str): Path to the image file to update
        metadata (dict): Metadata dictionary to serialize and store
        
    Returns:
        None
        
    Notes:
        - Updates 'TimesScanned' based on previous metadata
        - Handles encoding as UTF-8 and conversion to EXIF format
        - Logs success or failure of the operation
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.error("❌ Failed to open image %s: %s", image_path, e)
        return

    exif_bytes = image.info.get("exif")
    if exif_bytes:
        exif_dict = piexif.load(exif_bytes)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    user_comment_tag = piexif.ExifIFD.UserComment

    # Update TimesScanned based on previous metadata in UserComment
    prev_meta = read_exif(image_path) if image_path != "Image0.jpg" else {}
    if prev_meta and "TimesScanned" in prev_meta:
        metadata["TimesScanned"] = prev_meta["TimesScanned"] + 1
    else:
        metadata["TimesScanned"] = 1
    logger.info("🆕 Setting TimesScanned to %d for image %s", metadata["TimesScanned"], image_path)

    formatted_metadata = json.dumps(metadata, indent=4)
    encoded_metadata = formatted_metadata.encode('utf-8')
    exif_dict["Exif"][user_comment_tag] = encoded_metadata

    new_exif_bytes = piexif.dump(exif_dict)
    try:
        image.save(image_path, image.format if image.format else "jpeg", exif=new_exif_bytes)
        logger.info("✅ EXIF metadata written to %s", image_path)
    except Exception as e:
        logger.error("❌ Failed to save image with updated EXIF: %s", e)

# =============================================================================
# Metadata Extraction from YOLO Results
# =============================================================================

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
    if isinstance(orig_image, str):
        loaded_image = cv2.imread(orig_image)
        shape = list(loaded_image.shape[:2]) if loaded_image is not None else [0, 0]
    elif hasattr(orig_image, "shape"):
        shape = list(orig_image.shape[:2])
    else:
        shape = [0, 0]

    boxes = (results[0].boxes.xyxy.cpu().numpy()
             if results and results[0].boxes.xyxy is not None
             else np.array([]))
    image_path = results[0].path if hasattr(results[0], "path") and results[0].path else ""

    previous_metadata = {}
    if image_path != "Image0.jpg":
        try:
            previous_metadata = read_exif(image_path)
        except Exception as e:
            logger.error("❌ Error reading EXIF from %s: %s", image_path, e)
            previous_metadata = {}
    else:
        logger.warning("💡 No previous Scans to read, you inputed a numpy array")

    times_scanned = previous_metadata.get("TimesScanned", 0)
    times_scanned = times_scanned + 1 if times_scanned else 1

    message_value = "Muchas gracias por ejecutar la DEMO del projecto"
    # Retrieve additional details from YOLO results
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
    speed_data = results[0].speed if hasattr(results[0], "speed") else {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    # Get class names from results
    class_names = results[0].names if hasattr(results[0], "names") else {}
    
    

    if isinstance(class_names, dict):
        class_names = list(class_names.values())


    mode_value = class_names[0] if class_names else "obj"

    metadata = {
        "boxes_coordinates": {str(idx): box for idx, box in enumerate(boxes)},
        "orig_shape": shape,
        "speed": {"preprocess": speed_data.get("preprocess", 0.0), 
                  "inference": speed_data.get("inference", 0.0), 
                  "postprocess": speed_data.get("postprocess", 0.0)},
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
        "message": message_value
    }
    return metadata


# =============================================================================
# Image Annotation Functions
# =============================================================================

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

def clean_exif_metadata(image_path):
    '''
    Removes the info inside the UserComment tag of the EXIF metadata.
    
    Args:
        image_path (str): Path to the image file to clean
        
    Returns:
        None
        
    Notes:
        - Preserves other EXIF data, only removes UserComment content
        - Useful for resetting processing history before new detection
    '''
    metadata = read_exif(image_path)
    if metadata == {}:
        logger.warning("⚠️ No metadata found in the image: %s", image_path)
        return
    
    #rewrite the image without UserComment tag
    try:
        with Image.open(image_path) as image:
            exif_bytes = image.info.get("exif")
            if not exif_bytes:
                logger.warning("⚠️ No EXIF data found in %s", image_path)
                return
            else:
                exif_dict = piexif.load(exif_bytes)
                exif_dict["Exif"][piexif.ExifIFD.UserComment] = b""
                new_exif_bytes = piexif.dump(exif_dict)
                image.save(image_path, exif=new_exif_bytes)
                logger.info("✅ EXIF metadata cleaned from %s", image_path)
    except Exception as e:
        logger.error("❌ Failed to clean EXIF metadata from %s: %s", image_path, e)
    


# =============================================================================
# Brick Detection Function
# =============================================================================

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
    
    with console.status("[bold green]Loading image and model...") as status:
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
            console.print("[yellow]⚠️ No output folder provided. Results will not be saved.[/]")
        else:
            os.makedirs(output_folder, exist_ok=True)
        
        status.update("[bold green]Running detection...")
    
    if use_progress:
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
                    composite_path = os.path.join(output_folder, "composite_image.jpg")
                    cv2.imwrite(composite_path, annotated_image)
                    metadata["annotated_image_path"] = composite_path
                    logger.info(f"💾 Saved annotated image to: {composite_path}")
                    
                if save_json:
                    # Save metadata as JSON
                    json_path = os.path.join(output_folder, "metadata.json")
                    with open(json_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    metadata["json_results_path"] = json_path
                    logger.info(f"💾 Saved metadata to: {json_path}")
                    
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
                composite_path = os.path.join(output_folder, "composite_image.jpg")
                cv2.imwrite(composite_path, annotated_image)
                metadata["annotated_image_path"] = composite_path
                logger.info(f"💾 Saved annotated image to: {composite_path}")
                
            if save_json:
                # Save metadata as JSON
                json_path = os.path.join(output_folder, "metadata.json")
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                metadata["json_results_path"] = json_path
                logger.info(f"💾 Saved metadata to: {json_path}")
    
    # Check results safely after it's been defined
    if results and len(results) > 0:
        table = Table(title="Brick Detection Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Bricks detected", str(len(boxes_np)))
        table.add_row("Confidence threshold", f"{conf:.2f}")
        table.add_row("Processing time", f"{metadata['speed']['inference']:.3f}s")
        
        console.print(Panel(table, title="[bold]Detection Complete[/]", border_style="green"))
    
    try:
        if isinstance(image_input, str):
            # Always update EXIF for the original image file
            write_exif(image_input, metadata)
        
        # Additionally update EXIF for annotated images if we're saving them
        if save_annotated and metadata.get("annotated_image_path"):
            write_exif(metadata["annotated_image_path"], metadata)
    except Exception as e:
        logger.error(f"❌ Failed to write EXIF metadata: {e}")
        
    return {
        "orig_image": image,
        "annotated_image": annotated_image,
        "cropped_detections": cropped_detections,
        "metadata": metadata,
        "boxes": boxes_np,
        "status": "success"  # Add status for consistency with read_detection
    }

# =============================================================================
# Stud Detection and Dimension Classification
# =============================================================================

def classify_dimensions(results, orig_image, dimension_map=config["STUDS_TO_DIMENSIONS_MAP"]):
    """
    Classifies the brick dimension based on detected stud positions.
    
    Args:
        results (list): YOLO detection results for studs
        orig_image (np.ndarray): Original image
        dimension_map (dict, optional): Mapping from stud counts to brick dimensions
        
    Returns:
        Union[str, dict]: Either:
            - String with dimension classification (e.g., "2x4")
            - Dictionary with dimension and annotated image for complex cases
            
    Notes:
        - For simple cases, uses direct mapping from stud count to dimension
        - For complex cases with multiple possibilities:
          - Analyzes stud pattern using regression line
          - Determines if studs are in line (Nx1) or rectangular (NxM) pattern
          - Returns both dimension classification and visualization
    """
    studs = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else np.array([])
    
    num_studs = len(studs)
    valid_stud_counts = dimension_map.keys()
    if num_studs not in valid_stud_counts:
        logger.error(f"[ERROR] Deviant number of studs detected ({num_studs}). Returning 'Error'.")
        return "Error"

    if num_studs in valid_stud_counts and isinstance(dimension_map[num_studs], str):
        logger.info(f"[INFO] Detected {num_studs} studs. Dimension: {dimension_map[num_studs]}.")
        # type the dimension on the image
        annotated_image = results[0].plot(labels=False)  # original_image.copy()
        h, w = annotated_image.shape[:2]
        text = f"{dimension_map[num_studs]}"
        # Position in bottom right with margin
        font_size = 0.5
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        text_x = w - text_size[0] - 10
        text_y = h - 10
        cv2.putText(annotated_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_size, (0, 255, 0), 2)
        return {
            "dimension": dimension_map[num_studs],
            "annotated_image": annotated_image
        }
    if num_studs in valid_stud_counts and isinstance(dimension_map[num_studs], list):
        # Process centers and regression line:
        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in studs]
        box_sizes = [((x_max - x_min + y_max - y_min) / 2) for x_min, y_min, x_max, y_max in studs]
        xs, ys = zip(*centers)
        m, b = np.polyfit(xs, ys, 1)
        deviations = [abs(y - (m * x + b)) for x, y in centers]
        threshold = np.mean(box_sizes) / 2
        classification_aux = "Nx1" if max(deviations) < threshold else "Nx2"
        logger.info(f"[DEBUG] Detected {num_studs} studs. Classification: {classification_aux}.")
        print(dimension_map[num_studs])
        possible_dimensions = [dimension_map[num_studs][1] if classification_aux == "Nx1" else dimension_map[num_studs][0]]
        logger.info(f"[INFO] Classification: {classification_aux}. Final dimension: {possible_dimensions[0]}.")

        annotated_image = orig_image.copy()
        for x, y in centers:
            cv2.circle(annotated_image, (int(x), int(y)), 3, (0, 255, 0), -1)
        x1 = 0
        y1 = int(m * x1 + b)
        x2 = annotated_image.shape[1]
        y2 = int(m * x2 + b)
        cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add dimension text in bottom right corner with half size
        h, w = annotated_image.shape[:2]
        text = f"{possible_dimensions[0]}"
        font_size = 0.5
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        text_x = w - text_size[0] - 10
        text_y = h - 10
        cv2.putText(annotated_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_size, (0, 255, 0), 2)

        return {
            "dimension": possible_dimensions[0],
            "annotated_image": annotated_image
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
    # message if output folder is not provided
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

        annotated_image = results[0].plot(labels=False)

        # Classify dimensions based on detected studs
        dimension_info = classify_dimensions(results, orig_image)
        dimension_result = dimension_info.get("dimension", "Unknown")
        annotated_image = dimension_info.get("annotated_image", annotated_image)
        metadata = extract_metadata_from_yolo_result(results, orig_image)
        # Set the mode explicitly to "stud" to distinguish the detection type
        metadata["mode"] = "stud"
        metadata["dimension"] = dimension_result
        
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
        logger.error(f"❌ Failed to write EXIF metadata: {e}")
    
    # Save annotated image if requested
    if save_annotated:
        # Create and save the annotated image
        annotated_path = os.path.join(output_folder, "annotated_image.jpg")
        cv2.imwrite(annotated_path, annotated_image)
        metadata["annotated_image_path"] = annotated_path
        logger.info("💾 Annotated image saved at: %s", annotated_path)
        
        # Also write EXIF to annotated image if we saved it
        try:
            write_exif(annotated_path, metadata)
        except Exception as e:
            logger.error(f"❌ Failed to write EXIF metadata to annotated image: {e}")

    return {
        "orig_image": orig_image,
        "annotated_image": annotated_image,
        "dimension": metadata.get("dimension", "Unknown"),
        "metadata": metadata,
        "status": "success"
    }

def run_full_algorithm(image_path, save_annotated=False, output_folder="", force_rerun=False, logo=config["LOGO_NUMPY"], external_progress=None):
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
    # handle empty output folder
    if output_folder == "" and save_annotated:
        output_folder = os.path.join(os.getcwd(), "tests", "test_results")
        os.makedirs(output_folder, exist_ok=True)
        logger.info("📂 Output folder not provided. Saving results in: %s", output_folder)

    # If force_rerun is True, clean existing EXIF metadata
    if force_rerun:
        logger.info("🔄 Force rerun requested. Cleaning existing EXIF metadata.")
        clean_exif_metadata(image_path)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error("❌ Failed to load image from path: %s", image_path)
        return None

    # Use external progress if provided, otherwise create a new one
    if external_progress:
        # Use the provided progress context
        brick_results = detect_bricks(
            image_path,  # Use path for caching support
            save_annotated=False, 
            save_json=False, 
            output_folder=output_folder,
            use_progress=False,  # Don't create a new progress bar
            force_rerun=force_rerun  # Pass force_rerun parameter
        )
    else:
        # Create our own progress context
        brick_results = detect_bricks(
            image_path,  # Use path for caching support
            save_annotated=False, 
            save_json=False, 
            output_folder=output_folder,
            force_rerun=force_rerun  # Pass force_rerun parameter
        )

    if brick_results is None:
        logger.error("❌ Error during brick detection.")
        return None

    # Get cropped detections from brick detection
    cropped_detections = brick_results.get("cropped_detections", [])
    logger.info(f"🔍 Found {len(cropped_detections)} cropped detections.")
    studs_results = []
    
    # Process each detected brick
    if len(cropped_detections) > 1:
        for idx, crop in enumerate(cropped_detections):
            logger.info(f"🔍 Running studs detection on cropped detection number {idx + 1}.")
            # Use numpy arrays directly for crops (no caching)
            studs_result = detect_studs(crop, save_annotated=False, force_rerun=True)
            if studs_result is None:
                logger.error(f"❌ No studs detected in cropped detection number {idx + 1}.")
                continue
            studs_results.append(studs_result)
            
    elif len(cropped_detections) == 1:
        logger.info(f"🔍 Running studs detection on single cropped detection.")
        # Use numpy array directly for the crop (no caching)
        studs_result = detect_studs(cropped_detections[0], save_annotated=False, force_rerun=True)
        if studs_result is None:
            logger.error("❌ No studs detected in the single cropped detection.")
            return None
        studs_results = [studs_result]
    else:
        logger.warning("⚠️ No brick detected. Running studs detection on the original image.")
        studs_result = detect_studs(image_path, save_annotated=False, force_rerun=force_rerun)
        if studs_result is None:
            logger.error("❌ No studs detected in the original image.")
            return None
        studs_results = [studs_result]

    logger.info(f"🔍 Found {len(studs_results)} valid studs results. Classifying dimensions...")
    
    # Create the composite image
    base_image = brick_results.get("annotated_image")
    
    if len(studs_results) > 1:
        # For multiple studs results, create a horizontal stack of all annotated studs images
        annotated_studs_images = [result.get("annotated_image") for result in studs_results]
        areas = [image.shape[0] * image.shape[1] for image in annotated_studs_images]
        sorted_images = [image for _, image in sorted(zip(areas, annotated_studs_images), reverse=True)]
        
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
    else:
        # For a single studs result
        if studs_results and len(studs_results) > 0:
            studs_image = studs_results[0].get("annotated_image")
            
            # Resize studs image to match base image width
            h, w = studs_image.shape[:2]
            new_w = base_image.shape[1]
            new_h = int(h * new_w / w)
            resized_studs_image = cv2.resize(studs_image, (new_w, new_h))
            
            # Stack base image and studs image vertically
            composite_image = np.vstack((base_image, resized_studs_image))
        else:
            logger.warning("⚠️ No valid studs results to create composite image.")
            composite_image = base_image.copy()
    
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
    
    # Save annotated image if requested
    if save_annotated:
        annotated_path = os.path.join(output_folder, "fullyScannedImage.jpg")
        cv2.imwrite(annotated_path, composite_image)
        logger.info("💾 Annotated image saved at: %s", annotated_path)
        
        # Optional: Save metadata if needed
        # combined_metadata = {
        #    "brick_detection": brick_results.get("metadata", {}),
        #    "stud_detection": [result.get("metadata", {}) for result in studs_results],
        #    "timestamp": datetime.datetime.now().isoformat()
        # }
        # json_path = os.path.join(output_folder, "full_algorithm_results.json")
        # with open(json_path, "w") as f:
        #    json.dump(combined_metadata, f, indent=2)
        # logger.info("💾 Results metadata saved at: %s", json_path)

    return {
        "brick_results": brick_results,
        "studs_results": studs_results,
        "composite_image": composite_image
    }



# =============================================================================
# End of Model Utils Module
# =============================================================================
