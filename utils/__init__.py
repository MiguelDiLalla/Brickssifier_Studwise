"""
LEGO Bricks ML Vision Utilities Package

This package contains all utility modules for the LEGO Bricks ML Vision project,
providing core functionality for LEGO brick and stud detection, dimension classification,
and result visualization.

Modules:
  - config_utils: Configuration setup and resource loading
  - detection_utils: Brick and stud detection operations
  - exif_utils: EXIF metadata reading and writing
  - classification_utils: LEGO brick dimension classification
  - visualization_utils: Image annotation and visualization functions
  - pipeline_utils: End-to-end detection and classification workflows
  - metadata_utils: Result metadata extraction and processing

Author: Miguel DiLalla
"""

__version__ = "1.0.0"

# Import key functions from each module for convenient access
try:
    # Configuration utilities
    from .config_utils import setup_utils, config
    
    # Detection utilities
    from .detection_utils import detect_bricks, detect_studs
    
    # EXIF utilities
    from .exif_utils import read_exif, write_exif, clean_exif_metadata
    
    # Classification utilities
    from .classification_utils import classify_dimensions, analyze_stud_pattern, get_common_brick_dimensions
    
    # Visualization utilities
    from .visualization_utils import annotate_scanned_image, read_detection, create_composite_image
    
    # Pipeline utilities
    from .pipeline_utils import run_full_algorithm, batch_process
    
    # Metadata utilities
    from .metadata_utils import extract_metadata_from_yolo_result

    # Define what gets imported with "from utils import *"
    __all__ = [
        # Configuration
        'setup_utils', 'config',
        
        # Core detection
        'detect_bricks', 'detect_studs',
        
        # EXIF handling
        'read_exif', 'write_exif', 'clean_exif_metadata',
        
        # Classification
        'classify_dimensions', 'analyze_stud_pattern', 'get_common_brick_dimensions',
        
        # Visualization
        'annotate_scanned_image', 'read_detection', 'create_composite_image',
        
        # Full pipeline
        'run_full_algorithm', 'batch_process',
        
        # Metadata
        'extract_metadata_from_yolo_result'
    ]

except ImportError as e:
    import logging
    logging.warning(f"⚠️ Error importing utilities: {e}")
    logging.warning("Some functions may not be available. Please ensure all dependencies are installed.")
    __all__ = []

# Log package initialization
import logging
logger = logging.getLogger(__name__)
logger.info("📦 LEGO Bricks ML Vision utilities package initialized.")