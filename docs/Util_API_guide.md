# LEGO Bricks ML Vision Utilities API Guide

## batch_utils.py
Tools for batch processing of images for brick and stud detection.

### Functions
- `check_existing_annotations(image_paths, output_folder)`: Checks which images already have annotation files
  - Args: List of image paths, Path to annotation output folder
  - Returns: Tuple of (to_process, already_processed) image paths

- `analyze_existing_annotations(annotation_files, dimensions_map)`: Analyzes existing YOLO annotation files
  - Args: List of annotation files, Mapping of class IDs to dimensions
  - Returns: Statistics from analyzing annotations

- `process_batch_inference(input_folder, output_folder, dimensions_map, progress=None, image_paths=None, skip_errors=False, progress_table=None)`: 
  - Processes multiple images for brick detection and annotation
  - Handles progress tracking and error reporting
  - Returns processing statistics

- `display_batch_results(stats, console)`: Displays formatted batch processing results

## classification_utils.py
Handles brick dimension classification based on stud detection patterns.

### Functions
- `classify_dimensions(results, orig_image, dimension_map=None)`: Classifies brick dimensions from stud detections
  - Args: YOLO results, original image, optional dimension mapping
  - Returns: Dictionary with dimension classification and annotated image

- `analyze_stud_pattern(studs)`: Analyzes spatial arrangement of detected studs
  - Args: Array of bounding boxes
  - Returns: Pattern analysis including type (linear/grid) and regression parameters

- `predict_dimension_from_pattern(studs, dimension_map=None)`: Predicts brick dimensions from stud pattern
  - Args: Stud detections, optional dimension mapping
  - Returns: Predicted dimension string

- `estimate_brick_size(studs, pixel_to_mm=None)`: Estimates physical brick dimensions
  - Args: Stud detections, optional pixel to mm ratio
  - Returns: Size information in pixels and millimeters

## config_utils.py
Manages configuration and resource loading.

### Functions
- `setup_utils(repo_download=False)`: Initializes configuration dictionary
  - Args: Whether to download missing assets
  - Returns: Complete configuration dictionary

- `get_model_path(model_type)`: Gets path to specific model file
  - Args: Model type ('bricks' or 'studs')
  - Returns: Path to model file

- `get_project_root()`: Returns project root directory path

- `ensure_dir_exists(path)`: Creates directory if it doesn't exist

## data_utils.py
Dataset processing utilities.

### Functions
- `convert_keypoints_to_bboxes(args)`: Converts keypoint annotations to bounding boxes
  - Args: Command arguments with input/output paths
  - Returns: Conversion statistics

- `create_conversion_demo(json_path, output_folder)`: Creates visual demo of annotation conversion
  - Args: Path to LabelMe JSON, output directory
  - Returns: Success boolean

- `convert_labelme_to_yolo_format(json_data)`: Converts LabelMe annotations to YOLO format
  - Args: LabelMe JSON data
  - Returns: YOLO format annotation strings

## detection_utils.py
Core detection functionality using YOLO models.

### Functions
- `detect_bricks(image_input, model=None, conf=0.25, save_json=False, save_annotated=False, output_folder="", use_progress=True, force_rerun=False)`:
  - Performs brick detection with comprehensive options
  - Returns detection results with annotations and metadata

- `detect_studs(image_input, model=None, conf=0.25, save_annotated=False, output_folder="", force_rerun=False)`:
  - Performs stud detection within brick regions
  - Returns detection results with dimension classification

## exif_utils.py
EXIF metadata handling for storing detection results.

### Functions
- `read_exif(image_path, metadata_template=None)`: Reads EXIF metadata from image
  - Args: Image path, optional metadata template
  - Returns: Parsed metadata dictionary

- `write_exif(image_path, metadata)`: Writes metadata to image EXIF
  - Args: Image path, metadata dictionary

- `clean_exif_metadata(image_path)`: Removes existing metadata
  - Args: Image path

- `extract_bbox_data(metadata)`: Extracts bounding box data from metadata
  - Args: Metadata dictionary
  - Returns: List of bounding box information

## metadata_utils.py
Metadata extraction and processing.

### Functions
- `extract_metadata_from_yolo_result(results, orig_image)`: Extracts metadata from YOLO results
  - Args: YOLO results, original image
  - Returns: Structured metadata dictionary

- `format_detection_summary(metadata)`: Creates user-friendly summary
  - Args: Metadata dictionary
  - Returns: Formatted summary text

- `calculate_detection_metrics(metadata_list)`: Calculates aggregate metrics
  - Args: List of metadata dictionaries
  - Returns: Metrics dictionary

## rich_utils.py
Rich console interface utilities.

### Functions
- `create_progress()`: Creates Rich progress bar with custom formatting
  - Returns: Configured progress bar

- `ProgressContext`: Context manager for progress tracking
  - Creates and manages progress bar lifecycle

## text_utils.py
Text processing utilities.

### Functions
- `safe_emoji(emoji_str)`: Platform-safe emoji handling
  - Args: String containing emoji
  - Returns: Platform-appropriate string

## visualization_utils.py
Image visualization and annotation.

### Functions
- `annotate_scanned_image(image_path)`: Annotates image with metadata
  - Args: Image path
  - Returns: Annotated image

- `read_detection(image_path)`: Reads detection results from image
  - Args: Image path
  - Returns: Dictionary with detection results

- `create_composite_image(base_image, detection_images, logo=None)`: Creates composite visualization
  - Args: Base image, list of detection images, optional logo
  - Returns: Composite image

- `draw_detection_visualization(image, boxes, class_names=None, confidence_scores=None, color=(0, 255, 0))`:
  - Draws bounding boxes and labels on image
  - Returns annotated image

- `display_results_table(brick_results, studs_results)`: Displays formatted results table
