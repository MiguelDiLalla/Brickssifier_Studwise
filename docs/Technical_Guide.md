MiguelDiLalla: lets fill up the @Technical_Guide.md for the project. the attached files correspont the core of its functionalities. rememeber your instructions

GitHub Copilot: # LEGO Bricks ML Vision Technical Guide

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Installation & Setup](#installation--setup)
3. [Core Components](#core-components)
4. [ML Pipeline](#ml-pipeline)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Development Guidelines](#development-guidelines)
8. [Troubleshooting](#troubleshooting)

## Project Architecture

The LEGO Bricks ML Vision project follows a modular architecture with clearly separated components:

```
LEGO_Bricks_ML_Vision/
├── utils/                  # Core functionality modules
│   ├── model_utils.py      # ML inference and processing
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

### model_utils.py

The core ML functionality module with the following key capabilities:

- **Configuration management**: The `setup_utils()` function initializes all configurations
- **YOLO model wrappers**: Functions for model loading and inference
- **EXIF metadata handling**: Read/write metadata to image files with `read_exif()` and `write_exif()`
- **Detection algorithms**:
  - `detect_bricks()`: Identifies LEGO bricks in images
  - `detect_studs()`: Identifies studs on LEGO bricks
  - `classify_dimensions()`: Determines brick dimensions based on stud patterns
  - `run_full_algorithm()`: Combines detection and classification

### data_utils.py

Utilities for dataset preparation:

- **Annotation conversion**: Convert between LabelMe and YOLO formats
- **Keypoint handling**: Convert point annotations to bounding boxes
- **Visualization tools**: Display YOLO annotations on images

### train.py

The complete model training pipeline:

- **Hardware detection**: Auto-detects available hardware for training
- **Dataset preparation**: Unzipping, validation and splitting
- **Training configuration**: Configurable epochs, batch size, etc.
- **Training execution**: Uses YOLOv8 for model training
- **Results handling**: Automated zipping and sharing of results

## ML Pipeline

### Data Flow

1. **Raw Data**: JPEG images of LEGO bricks
2. **Dataset Preparation**: Conversion to YOLO format via `data_utils.py`
3. **Data Splitting**: Train/val/test split in `train.py`
4. **Training**: YOLOv8 model training with customized parameters
5. **Inference**: Brick detection → Stud detection → Dimension classification
6. **Results**: Annotated images and metadata output

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
from utils import model_utils

# Initialize configuration
config = model_utils.setup_utils()

# Detect bricks in an image
result = model_utils.detect_bricks(
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
from utils import model_utils

# Run the full pipeline
result = model_utils.run_full_algorithm(
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

### model_utils.py

#### Core Functions

```python
def setup_utils(repo_download=False)
```
Initializes configuration dictionary with all paths, models, and settings.
- **Parameters:**
  - `repo_download` (bool): Whether to download missing assets from repository
- **Returns:** Configuration dictionary

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

```python
def run_full_algorithm(image_path, save_annotated=False, output_folder="", force_ran=False, logo=None)
```
Runs the complete brick detection and dimension classification pipeline.
- **Parameters:**
  - `image_path`: Path to the image file
  - `save_annotated`: Whether to save annotated images
  - `output_folder`: Output directory
  - `force_ran`: Force re-detection even if cached
  - `logo`: Optional logo to add to output images
- **Returns:** Dictionary with brick detection, stud detection, and composite image

#### EXIF Functions

```python
def read_exif(image_path, TREE=None)
```
Reads EXIF metadata from an image file.
- **Parameters:**
  - `image_path`: Path to image file
  - `TREE`: Optional metadata template
- **Returns:** Dictionary with parsed metadata

```python
def write_exif(image_path, metadata)
```
Writes metadata to image file's EXIF UserComment tag.
- **Parameters:**
  - `image_path`: Path to image file
  - `metadata`: Dictionary with metadata to write

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
   - ML functionality goes in `utils/model_utils.py`
   - Dataset functions go in `utils/data_utils.py`
   - CLI commands go in `lego_cli.py`

2. **Logging standard**:
   - Use emoji markers
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
1. Check model paths in `model_utils.py`
2. Ensure models exist in `presentation/Models_DEMO/`
3. Try running with repository download enabled:
   ```python
   config = model_utils.setup_utils(repo_download=True)
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
   model_utils.clean_exif_metadata(image_path)
   ```
3. Verify that piexif library is properly installed

---

This Technical Guide serves as a comprehensive reference for developers working with the LEGO Bricks ML Vision project. For additional support or questions, please refer to project documentation or contact the project maintainer.