# LEGO Bricks ML Vision - Technical CLI Reference

## 1. Installation & Setup

### Dependencies
```
pytorch
ultralytics (YOLOv8)
opencv-python
pillow
rich
click
numpy
```

### Environment Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download models (will be auto-downloaded on first run)
4. Verify setup: `python lego_cli.py --help`

### Project Structure
```
LEGO_Bricks_ML_Vision/
├── models/                # ML model files
│   ├── Brick_Model.pt    # Brick detection model
│   └── Stud_Model.pt     # Stud detection model
├── utils/                 # Core utilities
├── configs/              # Configuration files
├── cache/                # Temporary processing files
├── logs/                 # Log outputs
└── results/              # Processing results
```

## 2. Core CLI Commands

### Detection Commands

#### `detect-bricks`
Detect LEGO bricks in images using trained model.

```
Usage: lego_cli.py detect-bricks [OPTIONS]

Options:
  --image PATH            Input image or directory [required]
  --output PATH          Output directory
  --conf FLOAT           Detection confidence threshold (default: 0.25)
  --save-annotated       Save visualization of detections (default: True)
  --save-json           Save detection results as JSON (default: False)
  --clean-exif          Remove EXIF metadata before processing (default: False)
```

Return Value:
```python
{
    "annotated_image": np.ndarray,  # OpenCV image with detections
    "metadata": {
        "boxes": List[List[float]],  # [x1,y1,x2,y2] bounding boxes
        "scores": List[float],       # Confidence scores
        "timestamp": str,            # Processing timestamp
        "settings": {                # Detection settings
            "confidence": float,
            "model_version": str,
            ...
        }
    }
}
```

#### `detect-studs`
Detect studs on LEGO bricks and classify dimensions.

```
Usage: lego_cli.py detect-studs [OPTIONS]

Options:
  --image PATH           Input image or directory [required]
  --output PATH         Output directory
  --conf FLOAT          Detection confidence threshold (default: 0.25)
  --save-annotated      Save visualization (default: True)
  --clean-exif         Clean EXIF metadata (default: False)
```

Return Value:
```python
{
    "annotated_image": np.ndarray,  # Image with stud detections
    "metadata": {
        "dimension": str,           # Classified brick dimension (e.g., "2x4")
        "stud_count": int,         # Number of detected studs
        "pattern_type": str,       # "linear" or "grid"
        "timestamp": str,          # Processing timestamp
        "studs": {
            "boxes": List[List[float]],  # Stud bounding boxes
            "scores": List[float],       # Stud confidence scores
            "pattern_analysis": {        # Detailed pattern info
                "regression_params": Tuple[float, float],
                "max_deviation": float,
                "threshold": float
            }
        }
    }
}
```

#### `infer`
Run complete detection pipeline - bricks, studs, and classification.

```
Usage: lego_cli.py infer [OPTIONS]

Options:
  --image PATH           Input image or directory [required]
  --output PATH         Output directory
  --save-annotated      Save visualization (default: True)
  --force-run          Force re-run even if cached (default: False)
```

Return Value:
```python
{
    "composite_image": np.ndarray,  # Final visualization combining all stages
}
```

#### `batch-inference`
Process multiple images in batch mode.

```
Usage: lego_cli.py batch-inference [OPTIONS]

Options:
  --input PATH          Input directory [required]
  --output PATH        Output directory
  --skip-errors       Continue on errors (default: False)
```

### Data Processing Commands

#### `data-processing labelme-to-yolo`
Convert LabelMe JSON annotations to YOLO format.

```
Usage: lego_cli.py data-processing labelme-to-yolo [OPTIONS]

Options:
  --input PATH          Input JSON folder [required]
  --output PATH        Output directory
  --clean             Clean output directory first (default: False)
```

#### `data-processing keypoints-to-bboxes`
Convert keypoint annotations to bounding boxes.

```
Usage: lego_cli.py data-processing keypoints-to-bboxes [OPTIONS]

Options:
  --input PATH          Input folder [required]
  --output PATH        Output directory
  --area-ratio FLOAT   Area ratio for boxes (default: 0.4)
  --clean             Clean output first (default: False)
```

### Metadata Commands

#### `metadata inspect`
View detailed EXIF metadata for images.

```
Usage: lego_cli.py metadata inspect IMAGE
```

#### `metadata clean-batch`
Clean metadata from multiple images.

```
Usage: lego_cli.py metadata clean-batch [OPTIONS] FOLDER

Options:
  --force    Skip confirmation prompt
```

### Utility Commands

#### `cleanup`
Manage temporary files and directories.

```
Usage: lego_cli.py cleanup [OPTIONS]

Options:
  --all           Clean everything
  --logs-only     Clean only logs
  --cache-only    Clean only cache
  --results-only  Clean only results
```

## 3. Function Reference

### Core Detection Pipeline
- `detect_bricks()`: Primary brick detection
- `detect_studs()`: Stud detection and pattern analysis
- `run_full_algorithm()`: Complete detection pipeline

Each detection command has a corresponding callback function that can be used programmatically:

```python
from lego_cli import detect_bricks_cmd, detect_studs_cmd, infer_cmd

# Brick detection with return value
result = detect_bricks_cmd.callback(
    image="image.jpg",
    output="results",
    conf=0.3,
    save_annotated=True,
    save_json=True,
    clean_exif=False
)

# Stud detection with return value
result = detect_studs_cmd.callback(
    image="brick.jpg",
    output="results",
    conf=0.25,
    save_annotated=True,
    clean_exif=False
)

# Full pipeline with return value
result = infer_cmd.callback(
    image="image.jpg",
    output="results",
    save_annotated=True,
    force_run=False
)
```

### Pattern Analysis
- `analyze_stud_pattern()`: Analyze stud arrangements
- `classify_dimensions()`: Determine brick dimensions

### Data Processing
- `convert_labelme_to_yolo()`: Annotation conversion
- `process_batch()`: Batch image processing

### Visualization
- `save_annotated_image()`: Save detection visualizations
- `create_results_visualization()`: Create result grids
- `display_batch_summary()`: Show processing statistics

### Integration Notes

- All image arrays are returned as NumPy arrays in BGR format (OpenCV)
- Metadata includes timestamps and settings for reproducibility
- Each command saves files to disk AND returns data structures
- Return values are standardized across batch processing
- Progress tracking is handled automatically for long operations

### Error Handling

Commands follow this error handling pattern:
1. Return None on critical failures
2. Include error details in metadata on partial failures
3. Log details to cli.log
4. Display rich formatted errors when possible

Example error handling:
```python
try:
    result = detect_bricks_cmd.callback(image="image.jpg")
    if result is None:
        print("Detection failed completely")
    elif not result["metadata"].get("boxes"):
        print("No detections found")
    else:
        boxes = result["metadata"]["boxes"]
        print(f"Found {len(boxes)} bricks")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 4. Configuration

### Default Values
- `CONF_THRESHOLD = 0.25`: Detection confidence threshold
- `IMG_SIZE = 640`: YOLO input image size
- `MAX_BATCH_SIZE = 16`: Batch processing limit
- `PATTERN_ERROR_THRESHOLD = 10`: Pattern classification threshold

### Model Paths
- Brick Model: `models/Brick_Model_best20250123_192838.pt`
- Stud Model: `models/Stud_Model_best20250124_170824.pt`

## 5. Output Formats

### Detection Results JSON
```json
{
  "boxes": [[x1, y1, x2, y2], ...],
  "scores": [0.98, ...],
  "labels": ["2x4", ...],
  "class_ids": [0, ...],
  "timestamp": "..."
}
```

### Pattern Analysis Results
```json
{
  "centers": [[cx, cy], ...],
  "pattern_type": "linear|grid|unknown",
  "regression_params": [slope, intercept],
  "max_deviation": float,
  "threshold": float
}
```

## 6. Best Practices

### Performance Optimization
- Pre-load models when processing multiple images
- Use batch processing for large datasets
- Clean up temporary files regularly

### Error Handling
- Use `--skip-errors` for batch processing
- Check logs for detailed error messages
- Validate image formats before processing

### Storage Management
- Use cleanup command regularly
- Monitor cache directory size
- Archive or delete old results

## 7. Troubleshooting

### Common Issues
1. Model loading errors:
   - Verify model files exist
   - Check CUDA availability
   - Ensure correct model versions

2. Image processing errors:
   - Validate image formats
   - Check image dimensions
   - Verify file permissions

3. Memory issues:
   - Reduce batch size
   - Clean cache directory
   - Monitor system resources

### Debug Mode
Enable debug logging:
```bash
lego_cli.py --debug [COMMAND]
```

### Log Files
- CLI logs: `logs/cli.log`
- Utility logs: `logs/utils.log`
- Error details and stack traces

## 8. Technical Details

### Return Value Formats

All return values follow these conventions:
- Images are NumPy arrays (BGR color space)
- Coordinates are in [x1,y1,x2,y2] format
- Metadata includes processing timestamp
- All floating point values use 32-bit precision
- Lists are regular Python lists for compatibility

### Path Handling

The CLI uses these path resolution rules:
- Relative paths are resolved from current directory
- Output paths are created if they don't exist
- Input paths must exist before command execution
- Path validation happens before processing starts

### Performance Notes

- Images are processed in memory when possible
- Large images may be automatically resized
- Batch operations process images sequentially
- Models are loaded once and reused
- Progress bars update every 100ms

## 9. Integration Examples

### Basic Script Integration
```python
from lego_cli import detect_bricks_cmd
import cv2

# Run detection
result = detect_bricks_cmd.callback(
    image="test.jpg",
    conf=0.3
)

# Use the results
if result:
    # Show detection visualization
    cv2.imshow("Detections", result["annotated_image"])
    cv2.waitKey(0)
    
    # Print detection info
    boxes = result["metadata"]["boxes"]
    scores = result["metadata"]["scores"]
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"Brick {i+1}: {box} (conf: {score:.2f})")
```

### Batch Processing
```python
import glob
from lego_cli import detect_studs_cmd

# Process multiple bricks
results = []
for img_path in glob.glob("bricks/*.jpg"):
    result = detect_studs_cmd.callback(
        image=img_path,
        save_annotated=True
    )
    if result:
        results.append({
            'path': img_path,
            'dimension': result["metadata"]["dimension"],
            'studs': result["metadata"]["stud_count"]
        })

# Analyze results
dimensions = [r["dimension"] for r in results]
print(f"Processed {len(results)} bricks")
print(f"Found dimensions: {set(dimensions)}")
```

### Full Pipeline with Progress
```python
from lego_cli import infer_cmd
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Processing images...", total=len(images))
    
    for img in images:
        result = infer_cmd.callback(
            image=img,
            save_annotated=True
        )
        if result:
            # Save the composite visualization
            cv2.imwrite(
                f"results/{img.stem}_analysis.jpg",
                result["composite_image"]
            )
        progress.advance(task)
```