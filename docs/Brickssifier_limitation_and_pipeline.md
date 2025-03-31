# 🧱 LEGO Brick Dimension Classification Pipeline
### 📐 From Image Input to Spatial Inference — Architecture, Limitations, and Design Logic

---

## ⚠️ System Limitations
Before exploring the pipeline mechanics, it's important to outline its current constraints:

1. **Limited Brick Types**: The system only supports a **predefined set** of basic LEGO bricks (e.g., 2x2, 1x4, 2x4, etc.).
2. **2D Stud-Based Analysis Only**: The model exclusively analyzes the **top face of the brick**, inferring only **length and width** based on the **stud layout**. It does **not detect height** and cannot distinguish between bricks, plates, or tiles.
3. **No Tile Recognition**: Bricks **must have visible studs**. Flat tiles without studs are **not supported**.
4. **Orientation Dependency**: Bricks must be **facing the camera**, with studs clearly visible. **Rotated, flipped, or angled bricks** (e.g., upside-down) will not be correctly classified.

---

## 🔁 Full Pipeline Overview
### Step-by-Step Breakdown of the Classification Process

---

### 1. 📂 Input Image
- The user provides an image path via the CLI (required interface).
- Internally, utility functions support direct image processing from **NumPy arrays**, allowing flexible integration in notebooks or custom scripts.
- The image must contain one or more **basic, top-facing LEGO bricks**.

---

### 2. 🔏 Metadata Fingerprint Check (EXIF)
- Upon receiving an image file, the system checks the **EXIF `UserComment` tag**:
  - If metadata is present, the pipeline **retrieves and reuses** prior inference results.
  - This avoids redundant processing and improves traceability.
- Metadata includes:
  - Inference timestamp
  - Model and system information
  - Initial detection results
  - GitHub repository link

---

### 3. 🎯 Model #1 — Brick Detection (`YOLOv8`)
- A finetuned, single-class **YOLOv8 model** detects LEGO bricks.
- Returns bounding boxes for each detected brick.

---

### 4. ✂️ Brick Cropping
- The pipeline crops the original image into **individual brick segments** based on bounding boxes.
- Each segment becomes a new input for stud-level analysis.

---

### 5. 🧠 Model #2 — Stud Detection (`YOLOv8`)
- A second YOLOv8 model, also finetuned for **stud detection**, is applied to each cropped brick image.
- Outputs:
  - Bounding boxes for each stud
  - Stud center coordinates (used in post-processing)

---

### 6. 🔢 Stud Counting
- The number of detected studs is matched against a predefined `dimension_map`.
  - If the count has **only one valid match**, return that result directly.
  - If there are **multiple possible dimensions** (e.g., 2x4 vs. 1x8), proceed to spatial pattern analysis.

---

### 7. 📐 Geometric Pattern Analysis
> Required when a stud count maps to multiple possible dimensions.

#### a. Extract Stud Centers
- Convert each stud bounding box into a center point `(x, y)`.

#### b. Perform Linear Regression
- Fit a line `y = mx + b` across the stud centers.

#### c. Calculate Deviation
- Measure how far each stud lies from the regression line.
- Compare the **maximum deviation** to a dynamic threshold (based on half the average stud size).

#### d. Determine Pattern Type
- If `max deviation < threshold`: the studs are **linearly aligned** (e.g., "1x8").
- If `max deviation ≥ threshold`: the layout is a **grid pattern** (e.g., "2x4").

#### e. Select Final Dimension
- Based on the detected pattern type, return the best-fitting option from the dimension map.

---

### 8. 🖼️ Annotated Image Generation
The result image is enriched with visual indicators:
- Green dots showing stud centers
- Regression line:
  - Blue if linear
  - Violet if grid, with optional error bars
- Final brick dimension label in the bottom-right corner

---

### 9. 🔏 Metadata Embedding (Output Image)
- The final annotated image is tagged with updated **EXIF metadata** in the `UserComment` field:
  - All fingerprint data from the original
  - **Final dimension classification**
  - Inference version, system info, and repo UUID
- This creates a **self-contained, traceable output**.

---

### 10. ✅ Output Delivery
- The CLI, notebook, or Streamlit interface receives:
  - A clean, annotated image
  - Rich visual overlays
  - Fingerprinted metadata for reproducibility and traceability

---

## 🧠 Final Notes
This system exemplifies a hybrid pipeline combining:
- **Deep learning** for detection (YOLOv8)
- **Classic geometry** for disambiguation (regression + thresholds)
- **Metadata engineering** for traceability
- **UX attention** via clear CLI, visual feedback, and modular code

It reflects a thoughtful approach to machine learning as a creative and structured design tool under real-world constraints.

