# 🧱 LEGO Bricks ML Vision

A computer vision portfolio project demonstrating ML-powered LEGO brick detection and classification.

![Project Logo](Project_Logo.png)

## 🎯 Project Overview

This project showcases a hybrid approach to LEGO brick recognition combining deep learning and classical computer vision techniques. It demonstrates:

- 🔍 **Brick Detection**: YOLOv8-based model to identify LEGO bricks in images
- 📍 **Stud Detection**: Specialized model for detecting individual studs on bricks
- 📐 **Dimension Classification**: Algorithm to determine brick dimensions from stud patterns
- 🖼️ **Rich Output**: Annotated images with embedded metadata and repository links

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV, Pillow (PIL)
- **CLI Interface**: Rich, Click
- **Data Processing**: Albumentations, NumPy
- **Metadata Handling**: EXIF UserComment tags
- **Development**: Python 3.10+, Git

## 📦 Project Structure

```
├── data/               # Dataset organization
├── models/            # Trained YOLOv8 models
├── utils/             # Core utilities
│   ├── data_utils.py      # Dataset processing
│   ├── detection_utils.py # ML inference
│   └── ...
├── docs/              # Comprehensive guides
├── notebooks/         # Development notebooks
└── lego_cli.py       # CLI interface
```

## 🚀 Key Features

### Training Pipeline
1. Dataset preparation and augmentation
2. YOLOv8 model training configuration
3. Automated training for brick/stud detection
4. Model evaluation and export

### Inference Pipeline
1. Input image processing
2. Brick detection and cropping
3. Stud detection within crops
4. Dimension classification via stud pattern
5. Rich output generation with metadata

## 💻 Getting Started

### Prerequisites
```bash
python -m pip install -r requirements.txt
```

### Basic Usage
```bash
# Run inference on an image
python lego_cli.py detect image.jpg

# Train a new model
python lego_cli.py train --config configs/train_config.yaml
```

## 📚 Documentation

- [Training Guide](docs/Training_API_Guide.md)
- [Utils Guide](docs/ProjectUtilsGuide.md)
- [CLI Guide](docs/Technical_CLI_Guide.md)
- [Project Story](docs/Storyline_Project_Storytell.md)

## 🎯 Project Goals

1. **Technical Demonstration**: Showcase ML/CV skills through a real-world application
2. **Learning Portfolio**: Document the journey from concept to implementation
3. **Code Quality**: Demonstrate software engineering best practices
4. **User Experience**: Create an engaging and accessible tool

## 📊 Technologies Learned

- YOLOv8 model training and optimization
- Hybrid ML approaches (deep learning + classical algorithms)
- CLI development with Rich/Click
- Professional documentation practices
- Data pipeline automation
- Testing and validation strategies

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Connect

Created by [Miguel Di Lalla](https://www.linkedin.com/in/MiguelDiLalla) - Feel free to connect!