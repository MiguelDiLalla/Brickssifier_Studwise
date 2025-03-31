
![Hero Banner](presentation/HeroBanner.jpg)

# 🧱 Project: Brickssifier_Studwise

> *It classifies bricks. Imperfectly. Passionately.*

A portfolio case study on teaching machines to identify LEGO bricks — blending deep learning with playful engineering, and showcasing rapid technical growth.

---

## 📌 Overview

This project is a self-contained demonstration of end-to-end ML development, focusing on LEGO brick detection and classification. It combines:

- Object detection (YOLOv8)
- Keypoint detection (studs)
- Custom pattern-based classification
- Rich CLI and Streamlit deployment

It’s also a personal milestone: a hands-on transition from hospitality to tech, executed with minimal resources and maximum curiosity.

---

## ⚡ Quick Links

- 🧠 [Technical Utility API Guide](docs/Util_API_guide.md)
- 💻 [Command Line Interface Guide](docs/Lego_CLI_Guide.md)
- 📖 [Project Story & Development Timeline](docs/Storyline_Project_Storytell.md)
- 📊 **[Streamlit DEMO (try it!)](https://placeholder.streamlit.app)**
  - ⚠️ May take up to 1–2 minutes to load (free-tier cold start)
- 🧪 **[Kaggle Project Page (coming soon)](https://kaggle.com/placeholder)**
  - Includes: notebooks for YOLOv8 training, evaluation, and the 3 datasets used.

---

## 🎯 Motivation

This project was born from a single question:

> *"If I can instinctively recognize a LEGO brick, can I teach a machine to do it too?"*

The answer led to the creation of a hybrid detection pipeline, extensive CLI tooling, and a public demo. It also serves as my first full-stack ML project — built from scratch, piece by piece.

---

## 📁 Repository Structure (Key Elements)

```
├── lego_cli.py            # Main CLI interface
├── utils/                 # Core utility modules
│   ├── detection_utils.py     # ML inference logic
│   ├── classification_utils.py # Stud-based dimension inference
│   └── ...
├── docs/                 # Project documentation
├── notebooks/            # YOLOv8 training notebooks (to be uploaded)
├── presentation/         # Hero image, visuals, QR frames
├── results/              # Annotated inference outputs
└── README.md
```

> 🗂️ Full reference in the [CLI Guide](docs/Lego_CLI_Guide.md) and [Utils API Guide](docs/Util_API_guide.md)

---

## 🧠 How It Works (Pipeline Summary)

1. Detect bricks in the image using a YOLOv8 model
2. Crop and analyze individual brick regions
3. Detect studs using a second YOLOv8 model
4. Analyze stud pattern with geometric logic
5. Predict brick dimensions and embed metadata

Includes optional EXIF tagging and visual output framing.

---

## 🧪 Training and Data

Model training and dataset curation were done on **Kaggle Notebooks**, using:
- 3 custom LEGO datasets (bricks, studs, and synthetic composites)
- Data augmentation with Albumentations
- Fine-tuning pretrained YOLOv8n models

📌 *Links to the notebooks and datasets will be added to the Kaggle project page.*

---

## 🖥️ Using the Project

### 🔧 CLI (Recommended)
```bash
# Detect a brick from image
python lego_cli.py infer --image input.jpg --save-annotated

# Run batch processing
python lego_cli.py batch-inference --input folder/ --output results/
```
More in [CLI Guide](docs/Lego_CLI_Guide.md).

### 🖼️ Streamlit App
Try it online: **[Streamlit DEMO](https://placeholder.streamlit.app)**

---

## 🧩 Development Timeline (Summary)

| Phase | Highlights |
|-------|-----------|
| **1. Ideation** | Inspired by childhood LEGO play & engineering instinct |
| **2. Dataset Prep** | 27 brick classes, ~2000 images, manual annotation |
| **3. Detection** | YOLOv8 fine-tuning for bricks and studs |
| **4. Classification** | Custom geometry algorithm based on stud layout |
| **5. CLI Design** | Fully documented with Rich-based UX |
| **6. Demo Deployment** | Streamlit frontend + EXIF metadata tagging |

Full story: [Project Story](docs/Storyline_Project_Storytell.md)

---

## ❤️ Thank You for Reading!

This project marks a transition point in my career and a personal victory in learning how to think like an engineer.

> If you enjoyed exploring this project or see potential in my work, don’t hesitate to reach out.

🔗 [Visit my website](https://migueldilalla.github.io/)  
💼 [Connect on LinkedIn](https://www.linkedin.com/in/MiguelDiLalla/)  
📨 Or open an issue / star the repo if you'd like to collaborate!

---

© Miguel Di Lalla — LEGO® is a trademark of the LEGO Group, which does not sponsor or endorse this project.

