"""
LEGACY MODEL UTILS MODULE - DEPRECATED

This module is kept for backward compatibility but is DEPRECATED.
Please use the specific utility modules instead:

- utils.config_utils: Configuration and setup
- utils.detection_utils: Brick and stud detection
- utils.exif_utils: EXIF metadata handling
- utils.classification_utils: Dimension classification
- utils.visualization_utils: Image annotation
- utils.pipeline_utils: End-to-end algorithms

Author: Miguel DiLalla
"""

import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Display deprecation warning on import
warnings.warn(
    "The module 'model_utils' is deprecated and will be removed in a future version. "
    "Please use the specific utility modules (config_utils, detection_utils, etc.) instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new modular structure for backward compatibility
from utils.config_utils import setup_utils, config
from utils.detection_utils import detect_bricks, detect_studs
from utils.exif_utils import read_exif, write_exif, clean_exif_metadata
from utils.classification_utils import classify_dimensions, analyze_stud_pattern, get_common_brick_dimensions
from utils.visualization_utils import annotate_scanned_image, read_detection, create_composite_image, draw_detection_visualization
from utils.pipeline_utils import run_full_algorithm, batch_process
from utils.metadata_utils import extract_metadata_from_yolo_result

# Log warning about deprecated usage
logger.warning("⚠️ Using deprecated 'model_utils' module. Please migrate to the new modular structure.")

# For better backward compatibility, create an alias for EXIF_METADATA_DEFINITIONS
EXIF_METADATA_DEFINITIONS = config["EXIF_METADATA_DEFINITIONS"]
STUDS_TO_DIMENSIONS_MAP = config["STUDS_TO_DIMENSIONS_MAP"]
LOADED_MODELS = config["LOADED_MODELS"]
LOGO_NUMPY = config["LOGO_NUMPY"]
TEST_IMAGES = config["TEST_IMAGES"]
