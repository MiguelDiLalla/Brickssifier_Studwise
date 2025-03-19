# LEGO Vision Project Utilities Guide

A comprehensive guide to using the monolithic `LegoProjectUtils.py` script for LEGO brick detection, analysis, and data processing.

## Overview

`LegoProjectUtils.py` is a standalone script that incorporates all core functionalities of the LEGO Bricks ML Vision project in a single, enhanced interface. It provides a unified command-line interface with rich visual feedback for all operations.

### Key Features

- **Brick Detection**: Identify LEGO bricks in images
- **Stud Detection**: Analyze stud patterns on bricks
- **Dimension Classification**: Determine brick dimensions
- **Batch Processing**: Handle multiple images efficiently
- **Data Processing**: Convert and validate annotations
- **Metadata Management**: Handle image EXIF data

## Installation

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Place the script in your project directory alongside the models folder.

## Command Reference

### Basic Detection Commands

#### Single Image Detection
```bash
# Basic brick detection
python LegoProjectUtils.py detect --image test.jpg --mode bricks

# Full analysis with custom confidence
python LegoProjectUtils.py detect --image test.jpg --conf 0.35

# Stud detection only
python LegoProjectUtils.py detect --image brick.jpg --mode studs
```

#### Options
- `--image`: Path to input image (required)
- `--mode`: Detection mode [`bricks`|`studs`|`full`] (default: `full`)
- `--conf`: Detection confidence threshold (0.0-1.0)
- `--output`: Custom output directory

### Batch Processing

```bash
# Process entire folder
python LegoProjectUtils.py batch --input data/images/

# With custom settings
python LegoProjectUtils.py batch --input data/images/ --conf 0.3 --output results/batch1/

# Continue on errors
python LegoProjectUtils.py batch --input data/images/ --skip-errors
```

#### Options
- `--input`: Input folder containing images (required)
- `--output`: Output folder for results
- `--conf`: Detection confidence threshold
- `--skip-errors`: Continue processing if individual images fail

### Data Conversion

#### LabelMe to YOLO Format
```bash
python LegoProjectUtils.py convert --input data/labelme/ --format labelme-to-yolo
```

#### Keypoints to Bounding Boxes
```bash
python LegoProjectUtils.py convert --input data/points/ --format keypoints-to-boxes --clean
```

#### Options
- `--input`: Input folder with annotation files (required)
- `--format`: Conversion format [`labelme-to-yolo`|`keypoints-to-boxes`] (required)
- `--output`: Output folder for converted annotations
- `--clean`: Clean output directory before conversion

### Metadata Management

#### Clean Single Image
```bash
python LegoProjectUtils.py clean-metadata image.jpg
```

#### Clean Batch
```bash
python LegoProjectUtils.py clean-metadata data/images/ --batch
```

## Output Structure

### Detection Results
```
output/
  ├── bricks/          # Brick detection results
  │   ├── detections.jpg
  │   └── metadata.json
  ├── studs/           # Stud detection results
  │   ├── brick_0.jpg
  │   └── brick_0_metadata.json
  └── summary.json     # Analysis summary
```

### Batch Processing Results
```
output_folder/
  ├── image1/
  │   ├── bricks/
  │   ├── studs/
  │   └── summary.json
  ├── image2/
  │   └── ...
  └── batch_summary.json
```

## Detection Pipeline

1. **Brick Detection**
   - Uses YOLOv8 model for brick detection
   - Generates bounding boxes and confidence scores
   - Saves annotated visualization

2. **Stud Detection** (for each detected brick)
   - Crops brick region from original image
   - Detects individual studs
   - Determines brick dimensions from stud pattern

3. **Results Processing**
   - Combines all detections
   - Generates summary statistics
   - Creates visualizations
   - Saves metadata

## Error Handling

The script uses a comprehensive error handling system with specific exceptions:

- `ValidationError`: Input validation failures
- `ProcessingError`: Image/data processing issues
- `ModelError`: ML model operation failures
- `MetadataError`: EXIF metadata handling issues

All errors are logged and provide clear feedback through the rich UI.

## Performance Considerations

- Large images are automatically resized for processing
- Memory usage scales with image size and batch size
- Consider breaking very large batches into smaller chunks
- Progress tracking available for all operations
- Caching system for intermediate results

## Logging

The script maintains detailed logs in:
```
logs/
  └── lego_vision.log
```

Enable debug output with:
```bash
python LegoProjectUtils.py --debug <command>
```

## Best Practices

1. **Image Preparation**
   - Use well-lit, clear images
   - Ensure consistent image quality
   - Clean EXIF metadata if needed

2. **Detection**
   - Start with default confidence (0.25)
   - Adjust based on results
   - Use full pipeline for best results

3. **Batch Processing**
   - Organize images in clean folders
   - Use `--skip-errors` for large batches
   - Monitor system resources

4. **Data Management**
   - Keep original files backed up
   - Use clean option carefully
   - Check conversion results

## Common Issues and Solutions

1. **Low Detection Accuracy**
   - Adjust confidence threshold
   - Check image quality
   - Ensure proper lighting

2. **Processing Errors**
   - Check image format compatibility
   - Verify file permissions
   - Monitor available memory

3. **Conversion Issues**
   - Verify input format compliance
   - Check file references
   - Ensure complete metadata

## Contributing

Feel free to contribute to this project by:
1. Reporting issues
2. Suggesting improvements
3. Submitting pull requests

## License

This project is licensed under MIT License. See LICENSE file for details.