# LEGO Bricks ML Vision Technical Guide

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Installation & Setup](#installation--setup)
3. [Core Components](#core-components)
4. [ML Pipeline](#ml-pipeline)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Development Guidelines](#development-guidelines)
8. [Troubleshooting](#troubleshooting)
9. [CLI Features](#cli-features)

## Project Architecture

The LEGO Bricks ML Vision project follows a modular architecture with clearly separated components:

```
LEGO_Bricks_ML_Vision/
├── utils/                  # Core functionality modules
│   ├── config_utils.py     # Configuration management
│   ├── detection_utils.py  # Brick and stud detection
│   ├── classification_utils.py # Dimension classification
│   ├── visualization_utils.py # Image annotation and display
│   ├── metadata_utils.py   # Detection metadata processing
│   ├── exif_utils.py       # EXIF metadata handlers
│   ├── pipeline_utils.py   # Full algorithm orchestration
│   ├── text_utils.py       # Text formatting utilities
│   ├── rich_utils.py       # Rich CLI interface components
│   └── data_utils.py       # Dataset processing utilities
├── train.py                # Training pipeline
├── lego_cli.py             # Command-line interface
├── tests/                  # Unit tests
├── presentation/           # Demo materials
│   ├── Models_DEMO/        # Pre-trained models
│   ├── Test_images/        # Test images
│   └── Datasets_Compress/  # Compressed datasets
└── results/                # Output directory
```

### Design Philosophy

The architecture follows these principles:
- **Modularity**: Specialized utility modules for each concern
- **Separation of concerns**: Clear distinction between ML functionality and user interface
- **Configuration over coding**: External configuration wherever possible
- **Rich logging**: Comprehensive logging with emoji markers for readability
- **Metadata persistence**: EXIF-based metadata for tracking processing history

## Installation & Setup

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LEGO_Bricks_ML_Vision.git
   cd LEGO_Bricks_ML_Vision
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python lego_cli.py --help
   ```

### Directory Structure Setup

The project automatically handles most directory creation, but you can manually run:

```bash
python lego_cli.py cleanup  # Reset to clean state
```

## Core Components

### config_utils.py

Core configuration module with the following key capabilities:
- **Configuration initialization**: The `setup_utils()` function initializes all configurations
- **Model loading**: Functions for loading YOLO models from disk or repository
- **Resource management**: Downloads assets from repository when needed
- **Path management**: Handles project directories and ensures paths exist

### detection_utils.py

Handles detection operations using YOLO models:
- **Brick detection**: Identifies LEGO bricks in images with `detect_bricks()`
- **Stud detection**: Identifies studs on LEGO bricks with `detect_studs()`
- **Result processing**: Extracts and processes detection information

### classification_utils.py

Utilities for dimension classification:
- **Dimension classification**: `classify_dimensions()` determines brick dimensions based on stud patterns
- **Pattern analysis**: `analyze_stud_pattern()` analyzes spatial arrangement of studs
- **Size estimation**: `estimate_brick_size()` calculates physical dimensions

### visualization_utils.py

Handles image annotation and visualization:
- **Image annotation**: `annotate_scanned_image()` draws detection results on images
- **Composite creation**: `create_composite_image()` combines multiple visualizations
- **Detection reading**: `read_detection()` extracts information from annotated images

### metadata_utils.py

Processes metadata from detections:
- **Metadata extraction**: `extract_metadata_from_yolo_result()` processes YOLO outputs
- **Summary formatting**: `format_detection_summary()` creates user-friendly summaries
- **Metrics calculation**: `calculate_detection_metrics()` computes aggregate statistics

### exif_utils.py

Handles EXIF metadata operations:
- **Reading metadata**: `read_exif()` extracts structured data from image EXIF
- **Writing metadata**: `write_exif()` stores detection results in image files
- **Metadata management**: Functions for cleaning and copying EXIF data

### pipeline_utils.py

Orchestrates the full detection pipeline:
- **Full algorithm**: `run_full_algorithm()` coordinates brick detection, stud detection, and classification
- **Batch processing**: `batch_process()` handles multiple images with progress tracking
- **Result aggregation**: Combines results from different detection stages

### data_utils.py

Utilities for dataset preparation:
- **Annotation conversion**: Convert between LabelMe and YOLO formats
- **Keypoint handling**: Convert point annotations to bounding boxes
- **Visualization tools**: Display YOLO annotations on images

## ML Pipeline

### Data Flow

1. **Raw Data**: JPEG images of LEGO bricks
2. **Dataset Preparation**: Conversion to YOLO format via `data_utils.py`
3. **Data Splitting**: Train/val/test split in `train.py`
4. **Training**: YOLOv8 model training with customized parameters
5. **Inference Pipeline**:
   - Brick detection: `detection_utils.detect_bricks()`
   - Stud detection: `detection_utils.detect_studs()`
   - Dimension classification: `classification_utils.classify_dimensions()`
   - Result visualization: `visualization_utils.create_composite_image()`
6. **Results**: Annotated images and metadata output via `exif_utils` and `metadata_utils`

### Model Architecture

The project uses two YOLOv8 models:

1. **Bricks Model**: Detects LEGO bricks in full images
   - Single class detection
   - Input: RGB images (various sizes)
   - Output: Bounding boxes with confidence scores

2. **Studs Model**: Detects studs on LEGO bricks
   - Single class detection
   - Input: Cropped brick images
   - Output: Stud positions used for dimension classification

### Dimension Classification Algorithm

The `classify_dimensions()` function uses:
1. Count-based classification for simple patterns (e.g., 1x1, 2x1)
2. Regression-based analysis for more complex cases:
   - Calculates a best-fit line through stud centers
   - Measures deviations from the line
   - Classifies based on stud arrangement pattern

## Usage Examples

### Basic Usage via CLI

```bash
# Train a brick detection model
python lego_cli.py train --mode bricks --epochs 20

# Run inference on an image
python lego_cli.py infer --images path/to/image.jpg

# Process dataset annotations
python lego_cli.py data-processing labelme-to-yolo --input path/to/json --output path/to/output

# Clean up temporary files
python lego_cli.py cleanup
```

### Python API Examples

Detecting bricks in an image:

```python
from utils import detection_utils, config_utils

# Initialize configuration
config = config_utils.config

# Detect bricks in an image
result = detection_utils.detect_bricks(
    "path/to/image.jpg",
    conf=0.25,
    save_annotated=True,
    output_folder="results/bricks"
)

# Access detection results
annotated_image = result["annotated_image"]
boxes = result["boxes"]
metadata = result["metadata"]
```

Running the full detection pipeline:

```python
from utils import pipeline_utils

# Run the full pipeline
result = pipeline_utils.run_full_algorithm(
    "path/to/image.jpg",
    save_annotated=True,
    output_folder="results/full"
)

# Access results
brick_results = result["brick_results"]
studs_results = result["studs_results"]
composite_image = result["composite_image"]

# Get brick dimensions
dimensions = studs_results[0]["dimension"]
```

## API Reference

### config_utils.py

```python
def setup_utils(repo_download=False)
```
Initializes configuration dictionary with all paths, models, and settings.
- **Parameters:**
  - `repo_download` (bool): Whether to download missing assets from repository
- **Returns:** Configuration dictionary

```python
def get_model_path(model_type)
```
Gets the path to a specific model.
- **Parameters:**
  - `model_type` (str): Type of model ('bricks' or 'studs')
- **Returns:** Path to the model file

```python
def ensure_dir_exists(path)
```
Ensures a directory exists, creating it if necessary.
- **Parameters:** Directory path
- **Returns:** The same path that was passed in

### detection_utils.py

```python
def detect_bricks(image_input, model=None, conf=0.25, save_json=False, save_annotated=False, output_folder="")
```
Performs brick detection using YOLO model.
- **Parameters:**
  - `image_input`: Image path (str) or numpy array
  - `model`: YOLO model instance (uses default from config if None)
  - `conf`: Confidence threshold
  - `save_json`: Whether to save detection metadata as JSON
  - `save_annotated`: Whether to save annotated image
  - `output_folder`: Directory to save outputs
- **Returns:** Dictionary with detection results

```python
def detect_studs(image_input, model=None, conf=0.25, save_annotated=False, output_folder="")
```
Detects studs in an image and classifies brick dimensions.
- **Parameters:** Similar to `detect_bricks()`
- **Returns:** Dictionary with stud detection results and dimension classification

### classification_utils.py

```python
def classify_dimensions(results, orig_image, dimension_map=None)
```
Classifies the brick dimension based on detected stud positions.
- **Parameters:**
  - `results`: YOLO detection results for studs
  - `orig_image`: Original image
  - `dimension_map`: Mapping from stud counts to brick dimensions
- **Returns:** Dictionary with dimension classification and annotated image

```python
def analyze_stud_pattern(studs)
```
Analyzes the pattern of detected studs to determine arrangement.
- **Parameters:** Array of bounding boxes
- **Returns:** Analysis result dictionary

### pipeline_utils.py

```python
def run_full_algorithm(image_path, save_annotated=False, output_folder="", force_rerun=False, logo=None, external_progress=None)
```
Runs the complete brick detection and dimension classification pipeline.
- **Parameters:**
  - `image_path`: Path to the image file
  - `save_annotated`: Whether to save annotated images
  - `output_folder`: Output directory
  - `force_rerun`: Force re-detection even if cached
  - `logo`: Optional logo to add to output images
  - `external_progress`: Optional external progress context from CLI
- **Returns:** Dictionary with brick detection, stud detection, and composite image

```python
def batch_process(image_paths, output_folder="batch_results", force_rerun=False)
```
Process multiple images in batch mode.
- **Parameters:**
  - `image_paths`: List of image paths to process
  - `output_folder`: Base folder for saving results
  - `force_rerun`: Whether to force re-detection even if cached
- **Returns:** Dictionary mapping image paths to result dictionaries

### exif_utils.py

```python
def read_exif(image_path, metadata_template=None)
```
Reads EXIF metadata from an image file.
- **Parameters:**
  - `image_path`: Path to image file
  - `metadata_template`: Optional metadata template
- **Returns:** Dictionary with parsed metadata

```python
def write_exif(image_path, metadata)
```
Writes metadata to image file's EXIF UserComment tag.
- **Parameters:**
  - `image_path`: Path to image file
  - `metadata`: Dictionary with metadata to write

```python
def clean_exif_metadata(image_path)
```
Removes the info inside the UserComment tag of the EXIF metadata.
- **Parameters:** Path to the image file

### visualization_utils.py

```python
def annotate_scanned_image(image_path)
```
Annotates an image with metadata from EXIF.
- **Parameters:** Path to the image file
- **Returns:** Annotated image with bounding boxes and project logo

```python
def read_detection(image_path)
```
Reads detection results from image's EXIF metadata.
- **Parameters:** Path to the image file
- **Returns:** Dictionary with detection information

```python
def create_composite_image(base_image, detection_images, logo=None)
```
Creates a composite image showing the base image with all detections.
- **Parameters:**
  - `base_image`: Original image with brick detections
  - `detection_images`: List of images showing stud detections
  - `logo`: Optional logo to add at the bottom
- **Returns:** Composite image with all visualizations

### metadata_utils.py

```python
def extract_metadata_from_yolo_result(results, orig_image)
```
Extracts metadata from YOLO results and original image.
- **Parameters:**
  - `results`: YOLO detection results
  - `orig_image`: Original image or path
- **Returns:** Structured metadata dictionary

```python
def format_detection_summary(metadata)
```
Creates a user-friendly summary from detection metadata.
- **Parameters:** Metadata dictionary
- **Returns:** Formatted summary text

### data_utils.py

#### Key Functions

```python
def convert_labelme_to_yolo(args)
```
Converts LabelMe JSON annotations to YOLO format.
- **Parameters:** Command-line arguments with input/output paths

```python
def convert_keypoints_to_bboxes(args)
```
Converts keypoints to bounding boxes.
- **Parameters:** Command-line arguments

```python
def visualize_yolo_annotation(args)
```
Displays YOLO annotations on images.
- **Parameters:** Command-line arguments

### train.py

#### Main Functions

```python
def train_model(dataset_path, model_path, device, epochs, batch_size)
```
Trains a YOLO model with specified parameters.
- **Parameters:**
  - `dataset_path`: Path to dataset
  - `model_path`: Path to model (or pre-trained model)
  - `device`: Training device (GPU/CPU)
  - `epochs`: Number of training epochs
  - `batch_size`: Batch size for training
- **Returns:** Path to training results

## Development Guidelines

### Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run a specific test:

```bash
python -m pytest tests/test_model_utils.py::TestModelUtilsLocal::test_detect_bricks_via_path
```

### Adding New Features

1. **Module placement**:
   - General configuration in `config_utils.py`
   - Detection logic in `detection_utils.py`
   - Classification in `classification_utils.py`
   - Visualization in `visualization_utils.py`
   - Metadata handling in `metadata_utils.py` and `exif_utils.py`
   - Pipeline orchestration in `pipeline_utils.py`
   - Dataset functions in `data_utils.py`
   - CLI commands in `lego_cli.py`

2. **Logging standard**:
   - Use emoji markers (see `text_utils.py` for platform-safe emoji)
   - Include appropriate log levels
   - Example: `logging.info("✅ Model loaded successfully.")`

3. **Testing**:
   - Add unit tests for new features
   - Update existing tests when changing functionality

### Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Add docstrings to all functions
- Include emoji markers in log messages

## Troubleshooting

### Common Issues

#### Model Loading Errors

**Problem**: `No model loaded` error when running detection.

**Solution**:
1. Check model paths in `config_utils.py`
2. Ensure models exist in `presentation/Models_DEMO/`
3. Try running with repository download enabled:
   ```python
   from utils.config_utils import setup_utils
   config = setup_utils(repo_download=True)
   ```

#### CUDA Out of Memory

**Problem**: GPU memory errors during training or inference.

**Solution**:
1. Reduce batch size
2. Use smaller model variant
3. Process images at lower resolution

#### Missing EXIF Metadata

**Problem**: No metadata found when reading EXIF.

**Solution**:
1. Check if the image is in a format that supports EXIF (JPEG)
2. Try cleaning EXIF data and reprocessing:
   ```python
   from utils.exif_utils import clean_exif_metadata
   clean_exif_metadata(image_path)
   ```
3. Verify that piexif library is properly installed

## CLI Features

### Command Structure
- Global commands
- Command groups
- Subcommands
- Common options

### Interactive Features
- Progress tracking
- Rich output formatting
- Confirmation prompts
- Error handling

### Command Groups Overview

1. **Detection Commands**
   - Purpose: Image processing and object detection
   - Commands: detect-bricks, detect-studs, infer
   - Common options and configurations

2. **Training Commands**
   - Purpose: Model training and optimization
   - Commands: train
   - Configuration options and parameters

3. **Data Processing Commands**
   - Purpose: Dataset preparation and conversion
   - Commands: labelme-to-yolo, keypoints-to-bboxes, visualize, demo-conversion
   - Input/output specifications

4. **Metadata Commands**
   - Purpose: Metadata management
   - Commands: inspect, clean-batch
   - Data handling options

5. **Utility Commands**
   - Purpose: System maintenance
   - Commands: cleanup, version
   - Management options

### Detection Commands

#### detect-bricks
```bash
lego_cli.py detect-bricks --input PATH --output PATH [--confidence FLOAT] [--clean-exif]
```
- `--input PATH`: Path to input image or folder
- `--output PATH`: Path to save detection results
- `--confidence FLOAT`: Confidence threshold for detection
- `--clean-exif`: Clean existing EXIF data

### Training Commands

#### train
```bash
lego_cli.py train --config PATH [--force-extract] [--use-pretrained]
```
- `--config PATH`: Path to training configuration file
- `--force-extract`: Force dataset re-extraction
- `--use-pretrained`: Use LEGO-trained model base

### Data Processing Commands

#### labelme-to-yolo
```bash
lego_cli.py labelme-to-yolo --input PATH --output PATH [--clean]
```
- `--input PATH`: Input folder containing LabelMe JSON files
- `--output PATH`: Output folder for YOLO .txt files
- `--clean`: Clean output directory before conversion

#### keypoints-to-bboxes
```bash
lego_cli.py keypoints-to-bboxes --input PATH --output PATH [--area-ratio FLOAT] [--clean]
```
- `--input PATH`: Input folder containing keypoints JSON files
- `--output PATH`: Output folder for bounding box JSON files
- `--area-ratio FLOAT`: Total area ratio for bounding boxes
- `--clean`: Clean output directory before conversion

#### visualize
```bash
lego_cli.py visualize --input PATH --labels PATH [--grid-size GRID]
```
- `--input PATH`: Path to a single image or folder of images
- `--labels PATH`: Folder containing YOLO .txt labels
- `--grid-size GRID`: Grid size for visualization (e.g., 3x4)

#### demo-conversion
```bash
lego_cli.py demo-conversion --input PATH --samples INT --output PATH
```
- `--input PATH`: Input folder with LabelMe JSONs
- `--samples INT`: Number of samples to process
- `--output PATH`: Output directory for demo

### Metadata Commands

#### inspect
```bash
lego_cli.py metadata inspect IMAGE
```
- Displays structured view of all metadata

#### clean-batch
```bash
lego_cli.py metadata clean-batch FOLDER [--force]
```
- `--force`: Skip confirmation prompt

### Utility Commands

#### cleanup
```bash
lego_cli.py cleanup
```
- Cleans up temporary files and directories

#### version
```bash
lego_cli.py version
```
- Displays:
  - Project version
  - Python version
  - CUDA/GPU information
  - Operating system details

### CLI Best Practices

1. **Command Usage**
   - Using appropriate confidence thresholds
   - Handling batch operations
   - Managing output directories

2. **Performance Optimization**
   - Batch processing recommendations
   - Resource management
   - Progress monitoring

3. **Error Recovery**
   - Common error scenarios
   - Troubleshooting steps
   - Fallback behaviors

### Rich Output Features

- Progress bars with spinners for long operations
- Status indicators for operations
- Tree views for metadata inspection
- Confirmation prompts for destructive operations

### Error Handling

- Missing rich package fallback behavior
- Image format compatibility issues
- Directory permission problems