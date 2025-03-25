# LEGO Bricks ML Vision CLI Guide

## Overview

The LEGO Bricks ML Vision CLI provides a comprehensive interface for detecting LEGO bricks, analyzing stud patterns, and processing datasets. This tool integrates all core functionalities of the project into an easy-to-use command line interface.

## Installation

The CLI is part of the main project. After cloning the repository and installing dependencies:

```bash
pip install -r requirements.txt
```

## Usage Modes

The CLI can be used in two ways:

1. Command Line Interface
   - Direct terminal usage with commands and options
   - Interactive progress display and rich output formatting
   - Results saved to output directories

2. Programmatic Interface
   - Import and use functions directly in Python scripts
   - Get structured return values as Python dictionaries
   - Integrate with other applications

## Command Structure

The CLI is organized into several command groups:

### Detection Commands
- `detect-bricks`: Detect LEGO bricks in images
- `detect-studs`: Detect studs on LEGO bricks
- `infer`: Run complete detection pipeline
- `batch-inference`: Process multiple images

### Data Processing Commands
- `data-processing labelme-to-yolo`: Convert LabelMe annotations to YOLO format
- `data-processing keypoints-to-bboxes`: Convert keypoints to bounding boxes
- `data-processing visualize`: Display annotations on images
- `data-processing demo-conversion`: Generate conversion pipeline demonstration

### Metadata Commands
- `metadata inspect`: View image metadata
- `metadata clean-batch`: Clean metadata from images

### Utility Commands
- `cleanup`: Manage temporary files
- `visualize-batch`: Create grid visualizations of batch results

## Command Line Usage Examples

### Basic Brick Detection
```bash
# Detect bricks in a single image
lego_cli.py detect-bricks --image path/to/image.jpg --conf 0.3

# Process a directory of images
lego_cli.py detect-bricks --image path/to/images/ --output results/detection
```

### Complete Analysis Pipeline
```bash
# Run full pipeline on an image
lego_cli.py infer --image path/to/image.jpg

# Process multiple images with visualization
lego_cli.py infer --image path/to/images/ --save-annotated
```

### Dataset Processing
```bash
# Convert LabelMe annotations to YOLO format
lego_cli.py data-processing labelme-to-yolo --input data/annotations

# Visualize annotations
lego_cli.py data-processing visualize --input images/ --labels labels/
```

### Batch Processing
```bash
# Run batch inference
lego_cli.py batch-inference --input dataset/images --output results/batch

# Create visualization grids
lego_cli.py visualize-batch metadata.json --samples 6
```

## Programmatic Usage

The CLI commands can be used programmatically in Python scripts to get structured return values:

### Brick Detection

```python
from lego_cli import detect_bricks_cmd

# Get detection results as a dictionary
result = detect_bricks_cmd.callback(
    image="path/to/image.jpg",
    output="results/detection",
    conf=0.3,
    save_annotated=True,
    save_json=False,
    clean_exif=False
)

# Access results
annotated_image = result["annotated_image"]  # OpenCV image with detections
metadata = result["metadata"]  # Detection metadata dictionary
```

### Stud Detection

```python
from lego_cli import detect_studs_cmd

# Get stud detection and classification results
result = detect_studs_cmd.callback(
    image="path/to/brick.jpg",
    output="results/studs",
    conf=0.25,
    save_annotated=True,
    clean_exif=False
)

# Access results
annotated_image = result["annotated_image"]  # Image with stud detections
metadata = result["metadata"]  # Includes dimension classification
```

### Full Pipeline

```python
from lego_cli import infer_cmd

# Run complete analysis pipeline
result = infer_cmd.callback(
    image="path/to/image.jpg",
    output="results/full",
    save_annotated=True,
    force_run=False
)

# Access results
composite_image = result["composite_image"]  # Final visualization image
```

### Return Value Structure

#### detect_bricks_cmd
```python
{
    "annotated_image": np.ndarray,  # OpenCV image with detections
    "metadata": {
        "boxes": [...],  # Bounding boxes
        "scores": [...],  # Confidence scores
        "timestamp": str,
        "settings": {...},
        ...
    }
}
```

#### detect_studs_cmd
```python
{
    "annotated_image": np.ndarray,  # Image with stud detections
    "metadata": {
        "dimension": str,  # Classified brick dimension
        "stud_count": int,
        "pattern_type": str,
        "timestamp": str,
        ...
    }
}
```

#### infer_cmd
```python
{
    "composite_image": np.ndarray,  # Final visualization combining all stages
}
```

## Command Reference

### detect-bricks
Detect LEGO bricks in images using the trained model.

Options:
- `--image PATH`: Input image or directory [required]
- `--output PATH`: Output directory
- `--conf FLOAT`: Detection confidence threshold (default: 0.25)
- `--save-annotated/--no-save-annotated`: Save visualization (default: True)
- `--save-json/--no-save-json`: Save detection results as JSON (default: False)
- `--clean-exif/--no-clean-exif`: Remove EXIF metadata (default: False)

### detect-studs
Detect studs on LEGO bricks and classify dimensions.

Options:
- `--image PATH`: Input image or directory [required]
- `--output PATH`: Output directory
- `--conf FLOAT`: Detection confidence threshold (default: 0.25)
- `--save-annotated/--no-save-annotated`: Save visualization (default: True)
- `--clean-exif/--no-clean-exif`: Remove EXIF metadata (default: False)

### infer
Run the complete detection and classification pipeline.

Options:
- `--image PATH`: Input image or directory [required]
- `--output PATH`: Output directory
- `--save-annotated/--no-save-annotated`: Save visualization (default: True)
- `--force-run/--no-force-run`: Force re-running detection (default: False)

## Output Structure

Results are organized in the following directories:

```
results/
├── brick_detection/      # Individual brick detection results
├── stud_detection/      # Stud detection results
├── full_inference/      # Complete pipeline results
├── batch_inference/     # Batch processing results
└── demo_conversion/     # Annotation conversion demonstrations

cache/
└── datasets/            # Processed dataset files

logs/
└── cli.log             # Command execution logs
```

## Error Handling

The CLI includes comprehensive error handling with:
- Descriptive error messages
- Progress tracking for long operations
- Safe cleanup of temporary files
- Validation of input paths and parameters

## Rich Output

When the `rich` package is installed, the CLI provides enhanced visual output:
- Colored progress bars
- Tables for results display
- Progress spinners for long operations
- Formatted error messages

Install rich for enhanced output:
```bash
pip install rich
```