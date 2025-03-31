## 🧠 Model Summary
This model is a custom-trained YOLOv8n object detector designed to identify individual LEGO bricks in cluttered real-world settings. It was trained from scratch on a manually annotated dataset of ~2000 images, featuring diverse lighting, backgrounds, and LEGO pieces.

The model outputs classic YOLO bounding boxes (`x, y, w, h, confidence, class`), and is intended as the first stage of a hybrid pipeline that later performs stud detection and geometric analysis.

**Architecture:** YOLOv8n  
**Training:** From scratch (no pretrained weights)  
**Training data:** 70/20/10 split (train/val/test) on original image dataset  
**Use case:** Object detection (single class — 'brick')

---

## 🛠️ Usage
```python
from ultralytics import YOLO

model = YOLO("lego_brick_detector.pt")
results = model("your_image.jpg")
results[0].boxes.xyxy  # Bounding box coordinates
```
- **Input:** Image (JPG/PNG), any resolution
- **Output:** List of bounding boxes with confidence scores

⚠️ **Limitations:** Performance drops on:
- Transparent or dark-colored bricks
- Images with high glare, extreme angles, or cluttered scenes

---

## 🧩 System
This model is part of a larger hybrid detection system. Its output (bounding boxes of bricks) is used to crop and forward each region to a second model that detects studs, followed by geometric reasoning to infer dimensions.

📦 Pipeline endpoint: [Studwise_Brickssifier Streamlit App](https://placeholder.streamlit.app)

---

## ⚙️ Implementation Requirements
- **Training platform:** Kaggle Notebooks (dual NVIDIA T4 GPUs)
- **Training time:** ~30 minutes
- **Epochs:** Max 50 (early stop ~20–30)
- **Framework:** Ultralytics YOLOv8
- **Augmentations:** Albumentations (mosaic, brightness/contrast, hue)
- **Model size:** < 6MB (.pt format)

---

## 🔧 Model Characteristics
- **Trained from scratch** (no pretraining)
- **No pruning, no quantization**
- Very lightweight, mobile-friendly

---

## 📊 Data Overview
- **Images:** ~2000 real-world LEGO scenes
- **Split:** 70/20/10 (train/val/test)
- **Annotations:** Manual bounding boxes via LabelMe
- **Augmentations:** Applied during training (color/geo variations)

---

## 📈 Evaluation Results
While no formal benchmarks were stored, model performance proved sufficient for the case study:
- Trained over multiple rounds with clear qualitative improvements
- Verified via inference results on validation and unseen images

---

## ⚖️ Fairness & Limitations
- **No demographic data involved**
- Model is not suitable for production or sensitive applications
- Known failure modes include:
  - Dark or transparent bricks
  - Modified top surfaces (tapering, textures, etc.)
  - Bricks flipped upside down

> This project was intended as a personal case study in vision model development, helping the author explore technical considerations around training, annotation, model debugging, and deployment.

---

## 🔗 Source & Contact
- 🔍 Project: [Studwise_Brickssifier on GitHub](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)
- 🌐 Web: [migueldilalla.github.io](https://migueldilalla.github.io)
- 📽️ Try the models: [Streamlit Demo App](https://placeholder.streamlit.app)

---

---

# 📦 YOLOv8n Model Card — LEGO Studs Detector

## 🔹 Subtitle:
YOLOv8n-based keypoint detector for locating LEGO studs on cropped brick images.

---

## 🧠 Model Summary
This YOLOv8n model detects the positions of LEGO studs by identifying small, circular bumps on cropped images of individual bricks. Though a bounding-box model, it mimics keypoint behavior by using center-marked annotations and custom square generation.

**Architecture:** YOLOv8n  
**Training:** From scratch  
**Training data:** ~4000 images, cropped from brick detection results  
**Use case:** Object detection (single class — 'stud')

---

## 🛠️ Usage
```python
from ultralytics import YOLO

model = YOLO("lego_stud_detector.pt")
results = model("cropped_brick.jpg")
```
- **Input:** Cropped brick image
- **Output:** Bounding boxes around studs

⚠️ **Limitations:**
- Misses studs on transparent bricks
- Fails on bricks with modified top faces (e.g., textures)

---

## 🧩 System
This model follows the brick detector in the broader **Studwise_Brickssifier** pipeline. Its output (stud locations) feeds into a geometry algorithm to infer brick dimensions (rows x columns).

---

## ⚙️ Implementation Requirements
- **Platform:** Kaggle (NVIDIA T4 GPU)
- **Training time:** ~30 mins
- **Framework:** Ultralytics YOLOv8 + Albumentations
- **Annotations:** LabelMe (keypoints → bounding boxes via custom script)
- **Model size:** ~6MB (.pt)

---

## 🔧 Model Characteristics
- Trained from scratch
- Lightweight
- Clever conversion from keypoints to square bounding boxes based on:
  - Total number of studs
  - 40% image area heuristic

---

## 📊 Data Overview
- Images: ~4000 cropped bricks
- Annotations: Keypoints turned into bounding boxes
- Technique: Dynamic sizing via script using per-image ratios
- Split: 70/20/10

---

## 📈 Evaluation Results
- Qualitative results strong on most flat-topped bricks
- Performance drops on modified or translucent parts

---

## ⚖️ Fairness & Limitations
- No demographic attributes involved
- Limited scope: personal learning project
- Not reliable on non-standard LEGO parts or inverted bricks

---

## 🔗 Source & Contact
- Project: [GitHub Repository](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)
- Streamlit App: [Launch Demo](https://placeholder.streamlit.app)
- Portfolio: [migueldilalla.github.io](https://migueldilalla.github.io)

