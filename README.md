# LEGO Bricks ML Vision 🧱👁️

A machine learning project for detecting and classifying LEGO bricks through computer vision.

## Project Overview

This project demonstrates a complete machine learning workflow for LEGO brick detection and classification:

1. **Brick Detection** 🔍: Identifying LEGO bricks in images using YOLOv8
2. **Stud Detection** 📍: Locating individual studs on detected bricks
3. **Dimension Classification** 📏: Determining brick dimensions (e.g., 2x4) based on stud patterns

## Key Features

- Complete ML pipeline from dataset preparation to inference
- Multiple detection models (bricks and studs)
- Command-line interface for all operations
- Rich metadata handling with EXIF storage
- Professional logging and error handling
- Visualization tools for model results

## Technical Components

- **Training Pipeline**: Custom training process for YOLO models
- **Inference Engine**: Optimized detection and classification system
- **Dataset Utilities**: Tools for converting annotations between formats
- **CLI Tools**: User-friendly command-line interface

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LEGO_Bricks_ML_Vision.git
cd LEGO_Bricks_ML_Vision

# Install dependencies
pip install -r requirements.txt
```

### Usage Examples

Train a model:
```bash
python lego_cli.py train --mode bricks --epochs 20
```

Run inference on an image:
```bash
python lego_cli.py infer --images path/to/image.jpg
```

Process dataset annotations:
```bash
python lego_cli.py data-processing labelme-to-yolo --input path/to/json --output path/to/output
```

## Project Structure

```
LEGO_Bricks_ML_Vision/
├── train.py                 # Training pipeline
├── lego_cli.py              # Command-line interface
├── utils/
│   ├── model_utils.py       # Core inference and processing
│   └── data_utils.py        # Dataset processing utilities
├── tests/                   # Unit tests
├── presentation/            # Demo materials and test images
│   ├── Models_DEMO/         # Pre-trained models
│   └── Test_images/         # Sample images
└── docs/                    # Documentation
```

## Portfolio Context

This project was created as a first portfolio project by an aspiring junior data scientist. Although focused on computer vision, it demonstrates a comprehensive understanding of the machine learning development process from data preparation through model deployment.

## License

[License information]
