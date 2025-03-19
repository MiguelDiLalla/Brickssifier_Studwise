# YOLO Model Testing UI Specification

## Overview

This document outlines the layout and functionality of the graphical user interface (GUI) for the YOLO Model Testing application. The interface is designed to allow users to drag and drop `.jpg` and `.pt` files, process them accordingly, and visualize relevant data in a structured manner.

---

## User Interface Layout

The GUI consists of four main panels arranged in a structured manner:

```
+----------------------------------------------------+
|  Model Test: YOLO                                 |
|----------------------------------------------------|
|  Original Image      |  Annotated Image           |
|  [Image Display]     |  [Processed Image Display] |
|                      |                            |
|----------------------------------------------------|
|  Model Info          |  Metadata Dict             |
|  [Text Display]      |  { Key-Value Pairs }       |
+----------------------------------------------------+
```

### **Panel Descriptions:**

1. **Original Image Panel** (Top Left)
   - Displays the original `.jpg` image after being loaded.
   - The image is internally resized while maintaining aspect ratio to optimally fit the latest loaded model.
   - Ensures file integrity by verifying that the image is not corrupted before processing.

2. **Model Metadata Panel** (Bottom Left)
   - Displays metadata when a `.pt` file is loaded.
   - The application validates the `.pt` file to ensure it is a YOLO Ultralytics model.
   - Extracts and displays class names and other relevant information from the model.

3. **Annotated Image Panel** (Top Right)
   - Displays the processed image with model-detected annotations.
   - This section visualizes the object detection results performed by the most recently loaded model.
   - Draws bounding boxes and overlays detected class labels.

4. **Image Metadata Panel** (Bottom Right)
   - Displays extracted metadata from the image file.
   - Includes properties such as image dimensions, format details, and additional information relevant to processing.

---

## **Functionality**

- **Drag-and-Drop Support**
  - Users can drag and drop `.jpg` or `.pt` files into the interface.
  - Only one file type can be loaded at a time; attempting to load both types simultaneously results in an error message.

- **File Validation**
  - `.pt` files are validated to confirm they are trained YOLO Ultralytics models.
  - `.jpg` files are checked for corruption before processing.

- **Automatic Image Resizing**
  - When a `.jpg` file is loaded, it is resized while maintaining its aspect ratio to fit the most recent YOLO model.

- **Model Information Extraction**
  - For `.pt` files, class names and other important model attributes are extracted and displayed.

- **Processing Pipeline**
  - The core engine for detection and processing comes from the `lego_cli.py` script.
  - The interface interacts with this script to load models, process images, and display results.

---

## **Dependencies**

- **Ultralytics YOLO** for model validation and inference.
- **OpenCV** for image processing.
- **PyQt5/Tkinter (or another GUI framework)** for creating the interface.

---

## **Future Enhancements**

- Allow users to select different trained models from a dropdown menu.
- Provide real-time updates when an image is processed.
- Export detection results as JSON or text for further analysis.

---

This UI provides a streamlined and efficient way to test YOLO models and visualize results in a structured manner.

