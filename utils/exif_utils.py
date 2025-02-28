"""
EXIF Metadata Utilities for LEGO Bricks ML Vision

This module handles reading and writing EXIF metadata to images,
allowing for persistent storage of detection results and settings.

Key features:
  - Reading detection metadata from image EXIF
  - Writing structured metadata to image EXIF
  - Tracking detection history with scan counts
  - Cleaning metadata for fresh analysis

Author: Miguel DiLalla
"""

import logging
import json
import piexif
from PIL import Image

# Set up logging with emoji markers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("🚀 EXIF Utils module loaded.")

# Import project modules
from utils.config_utils import config

def read_exif(image_path, metadata_template=None):
    """
    Reads EXIF metadata from an image file and logs scan status.
    
    Args:
        image_path (str): Path to the image file to read metadata from
        metadata_template (dict, optional): Dictionary structure for default values if metadata is missing
            Defaults to EXIF_METADATA_DEFINITIONS from config.
            
    Returns:
        dict: Parsed metadata or empty dictionary if no metadata found
    
    Notes:
        - Handles special case for "image0.jpg" as numpy array source
        - Updates TimesScanned for tracking processing history
    """
    # Use default EXIF DEFINITIONS from CONFIG if template is None
    if "image0.jpg" in image_path:
        logger.info("🆕 Using numpy array source.")
        return {}

    if metadata_template is None:
        metadata_template = config["EXIF_METADATA_DEFINITIONS"]

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
        # Ensure defaults from template are present
        for key, default in metadata_template.items():
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
    prev_meta = read_exif(image_path) if image_path != "image0.jpg" else {}
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

def clean_exif_metadata(image_path):
    """
    Removes the info inside the UserComment tag of the EXIF metadata.
    
    Args:
        image_path (str): Path to the image file to clean
        
    Returns:
        None
        
    Notes:
        - Preserves other EXIF data, only removes UserComment content
        - Useful for resetting processing history before new detection
    """
    metadata = read_exif(image_path)
    if metadata == {}:
        logger.warning("⚠️ No metadata found in the image: %s", image_path)
        return
    
    # Rewrite the image without UserComment tag
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

def copy_exif(source_path, destination_path):
    """
    Copies EXIF data from a source image to a destination image.
    
    Args:
        source_path (str): Path to the source image
        destination_path (str): Path to the destination image
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Read metadata from source image
        metadata = read_exif(source_path)
        if not metadata:
            logger.warning("⚠️ No metadata to copy from %s", source_path)
            return False
        
        # Write to destination image
        write_exif(destination_path, metadata)
        logger.info("✅ EXIF metadata copied from %s to %s", source_path, destination_path)
        return True
    except Exception as e:
        logger.error("❌ Failed to copy EXIF metadata: %s", e)
        return False

def extract_bbox_data(metadata):
    """
    Extracts bounding box data from metadata in a usable format.
    
    Args:
        metadata (dict): Metadata dictionary from read_exif
        
    Returns:
        list: List of dictionaries with bounding box information
    """
    boxes_data = []
    
    if not metadata or "boxes_coordinates" not in metadata:
        logger.warning("⚠️ No bounding box data found in metadata")
        return boxes_data
    
    for box_id, box_info in metadata["boxes_coordinates"].items():
        if isinstance(box_info, dict) and "coordinates" in box_info:
            box_data = {
                "id": box_id,
                "coordinates": box_info["coordinates"],
                "confidence": box_info.get("confidence", None),
                "class": box_info.get("class", None)
            }
            boxes_data.append(box_data)
        elif isinstance(box_info, list) and len(box_info) >= 4:
            # Legacy format
            box_data = {
                "id": box_id,
                "coordinates": box_info[:4],
                "confidence": None,
                "class": None
            }
            boxes_data.append(box_data)
    
    return boxes_data
