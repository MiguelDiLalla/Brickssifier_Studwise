# Model Summary
This model is a custom-trained YOLOv8n object detector designed to identify individual LEGO bricks in cluttered real-world settings. It was trained from scratch on a manually annotated dataset of ~2000 images, featuring diverse lighting, backgrounds, and LEGO pieces. The model outputs classic YOLO bounding boxes (`x, y, w, h, confidence, class`), and is intended as the first stage of a hybrid pipeline that later performs stud detection and geometric analysis.

# Usage
```python
from ultralytics import YOLO

model = YOLO("lego_brick_detector.pt")
results = model("your_image.jpg")
results[0].boxes.xyxy  # Bounding box coordinates
```
- Input: Image (JPG/PNG), any resolution  
- Output: List of bounding boxes with confidence scores  
- Limitations: Lower accuracy with transparent, dark-colored, or unusually shaped bricks; struggles with extreme glare or cluttered scenes.

# System
This model is part of a broader hybrid detection system ("Studwise_Brickssifier") where the output bounding boxes are used to crop bricks and feed them into a secondary model for stud detection and shape classification.

# Implementation requirements
- Training platform: Kaggle Notebooks (dual NVIDIA T4 GPUs)  
- Training time: ~30 minutes  
- Epochs: Up to 50 (early stopping ~20–30)  
- Framework: Ultralytics YOLOv8  
- Augmentations: Albumentations (mosaic, brightness/contrast, hue)  
- Model size: < 6MB (.pt)

# Model Characteristics
## Model initialization
Trained from scratch (no pretraining).

## Model stats
Lightweight YOLOv8n architecture (~6MB), suitable for fast inference.

## Other details
Model was not pruned or quantized. No differential privacy techniques were applied.

# Data Overview
## Training data
Manually annotated photos of LEGO bricks placed on diverse surfaces. Total ~2000 images. Labeled via LabelMe with bounding boxes.

## Demographic groups
No human or demographic data involved.

## Evaluation data
Split used: 70% training, 20% validation, 10% test. Images share similar distribution.

# Evaluation Results
## Summary
Qualitative evaluation via inference over held-out and new images. Observed clear improvements across training runs.

## Subgroup evaluation results
No formal subgroup analysis. Known weaknesses include misclassification on transparent or unusually shaped bricks.

## Fairness
No fairness metrics apply. Dataset contains no personal or demographic data.

## Usage limitations
Not suitable for production, transparency-sensitive parts, or pieces with heavy texture/tapering.

## Ethics
This project is an educational case study. The author considered performance, annotation bias, and model limitations as part of the learning process.

