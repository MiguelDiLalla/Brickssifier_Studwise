"""
Configuration Utilities for LEGO Bricks ML Vision

This module handles configuration management, resource loading,
and project setup for the LEGO Bricks ML Vision project.

Key features:
  - Configuration initialization and management
  - Model loading and resource management
  - Path and environment setup
  - Repository asset downloading

Author: Miguel DiLalla
"""

import logging
import os
import base64
import json
import requests
import platform
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# Set up logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 Config Utils module loaded.")

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
        "orig_shape": [0, 0],          # Original image dimensions.
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
        "Repository": CONFIG_DICT["REPO_URL"],  # Repository URL.
        "message": ""                  # Custom message.
    }
    
    return CONFIG_DICT

# Initialize the global configuration
config = setup_utils()

def get_model_path(model_type):
    """
    Get the path to a specific model.
    
    Args:
        model_type (str): Type of model ('bricks' or 'studs')
        
    Returns:
        str: Path to the model file
    """
    if model_type not in config.get("MODELS_PATHS", {}):
        logger.error(f"❌ Unknown model type: {model_type}")
        return None
    
    relative_path = config["MODELS_PATHS"][model_type]
    abs_path = os.path.join(os.getcwd(), relative_path)
    
    if not os.path.exists(abs_path):
        logger.error(f"❌ Model file not found: {abs_path}")
        return None
    
    return abs_path

def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
        Path: Path object representing the project root
    """
    return Path(os.getcwd())

def ensure_dir_exists(path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path (str): Directory path
        
    Returns:
        str: The same path that was passed in
    """
    os.makedirs(path, exist_ok=True)
    return path
