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
    # Find most recent model files dynamically
    models_dir = os.path.join(os.getcwd(), "presentation", "Models_DEMO")
    
    def get_latest_model(substring):
        if not os.path.exists(models_dir):
            logger.warning(f"⚠️ Models directory not found: {models_dir}")
            return None
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt') and substring.lower() in f.lower()]
        if not model_files:
            logger.warning(f"⚠️ No {substring} model files found")
            return None
        latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
        return os.path.join("presentation", "Models_DEMO", latest_model)

    CONFIG_DICT["MODELS_PATHS"] = {
        "bricks": get_latest_model("Brick"),
        "studs": get_latest_model("Stud"),
        "multiclass": get_latest_model("Multi")
    }

    # Log found model paths
    for model_type, path in CONFIG_DICT["MODELS_PATHS"].items():
        if path:
            logger.info(f"✅ Found latest {model_type} model: {path}")
        else:
            logger.error(f"❌ No {model_type} model found")
    CONFIG_DICT["TEST_IMAGES_FOLDERS"] = {
        "bricks": r"presentation/Test_images/BricksPics",
        "studs": r"presentation/Test_images/StudsPics"
    }
    
    # Find the LEGO_Bricks_ML_Vision directory, case insensitive
    current_dir = os.getcwd()
    target_dir_variants = ["LEGO_Bricks_ML_Vision", "lego_bricks_ml_vision"]
    
    if os.path.basename(current_dir).lower() not in [d.lower() for d in target_dir_variants]:
        found = False
        # Check current directory
        for variant in target_dir_variants:
            potential_path = os.path.join(current_dir, variant)
            if os.path.exists(potential_path):
                os.chdir(potential_path)
                logger.info(f"✅ Changed working directory to: {potential_path}")
                found = True
                break
                
        if not found:
            # Check one level up
            parent_dir = os.path.dirname(current_dir)
            for variant in target_dir_variants:
                potential_path = os.path.join(parent_dir, variant)
                if os.path.exists(potential_path):
                    os.chdir(potential_path)
                    logger.info(f"✅ Changed working directory to: {potential_path}")
                    found = True
                    break
                    
        if not found:
            logger.warning("⚠️ Could not find LEGO Bricks ML Vision directory")
    
    CONFIG_DICT["WORKING_DIR"] = os.getcwd()
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
    # Scan all possible dimensions and create a dictionary mapping IDs to classes
    CONFIG_DICT["BRICKS_DIMENSIONS_CLASSES"] = {}
    class_id = 0
    for key, value in CONFIG_DICT["STUDS_TO_DIMENSIONS_MAP"].items():
        if isinstance(value, list):
            for v in value:
                CONFIG_DICT["BRICKS_DIMENSIONS_CLASSES"][class_id] = v
                class_id += 1
        else:
            CONFIG_DICT["BRICKS_DIMENSIONS_CLASSES"][class_id] = value
            class_id += 1
    # Log available classes
    logger.info("🧱 Available classes: %s", CONFIG_DICT["BRICKS_DIMENSIONS_CLASSES"])
    logger.info("🔢 Total number of classes: %d", len(CONFIG_DICT["BRICKS_DIMENSIONS_CLASSES"]))

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
