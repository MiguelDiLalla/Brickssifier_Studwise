# 🚀 **From Hospitality to Machine Learning: A Case Study on LEGO Bricks ML Vision**

## **🔷 ACT 1: The WHY (Hook & Context)**

### **A Story of Transition and Reinvention**
Miguel Di Lalla’s journey into **machine learning and AI** didn’t start in a university lab—it began in the fast-paced world of hospitality. Working in the service industry **taught him resilience, adaptability, and the reality of burnout.** The relentless pace and lack of long-term planning opportunities pushed him to rethink his future.

But problem-solving was always part of his DNA. From childhood, he spent countless hours building intricate LEGO structures, fascinated by how pieces fit together. Years later, that same **engineering mindset** would drive him to develop an AI model capable of **doing what he had instinctively done as a LEGO player for years: quickly identifying and sorting bricks.**

### **The Philosophy Behind the Project**
Miguel doesn’t see AI as just another tool—he sees it as an **amplifier of human potential.**

💡 *“AI will supercharge humanity. That’s the ideal. And for that to happen, we must learn to teach it.”*

That core belief led to the question that started everything:

🔥 *“If I, as a human, can instantly recognize LEGO bricks from a pile, can I teach a machine to do the same?”*

This simple yet powerful idea set the foundation for **LEGO Bricks ML Vision**, a project designed to push the boundaries of object detection and classification while serving as Miguel’s first fully independent machine learning case study.

---

## **🔷 ACT 2: The HOW (Technical Journey & Problem Solving)**

### **Facing Reality: Why a Single Model Wasn’t Enough**
Initially, Miguel set out to create a deep learning model that could take an image of a LEGO brick and return its precise dimensions. However, reality hit hard: **models trained from scratch weren’t generalizing well, and distinguishing similar bricks numerically was far more complex than anticipated.**

Rather than force a failing approach, **he pivoted.**

🛠️ **Breaking the problem into smaller, solvable tasks:**
1️⃣ **Detect LEGO bricks in the image** (Bounding Box Detection)
2️⃣ **Crop and isolate the detected bricks**
3️⃣ **Detect individual studs within the cropped images** (Keypoint Detection)
4️⃣ **Use an algorithm to classify the brick based on its stud count and layout**

This shift in strategy led Miguel to experiment with various frameworks—**Google Mediapipe, Detectron**, and finally, **YOLOv8**, which provided the best tradeoff between speed and ease of use.

### **Beyond Machine Learning: When Classic Algorithms Make the Difference**
While YOLO worked well for detection, **classification remained a challenge.** Certain bricks (e.g., a 2x4 and a 1x8) had the same stud count but needed to be differentiated. The solution? **A custom algorithm that treats the detected studs as points in a Cartesian plane, applying regression analysis to determine alignment and shape.**

This hybrid approach—**leveraging both ML models and traditional computational techniques**—not only improved accuracy but also showcased Miguel’s **engineering mindset** and ability to integrate multiple paradigms into a cohesive solution.

> 📌 **All technical details, including dataset handling, annotation methods, and algorithmic breakdowns, are extensively documented in the project’s official corpus.** Available at:  
> 🔗 **[LEGO Bricks ML Vision Repository](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)**

### **Building with the End User in Mind: The CLI Experience**
A good project isn’t just about results—it’s about usability. Understanding this, Miguel **designed a visually appealing Command Line Interface (CLI)** using **Rich**, ensuring:
✅ **Extensive documentation** within `--help`
✅ **Aesthetic execution** with emojis and color-coded outputs
✅ **A seamless workflow** to run detection, classification, and demos in one command

> 🎨 *“I’m an artist as well as an engineer. Aesthetics matter to me. My CLI isn’t just functional—it’s satisfying to use.”*

---

## **🔷 ACT 3: The IMPACT (Results & What’s Next)**

### **The Breakthrough Moment: When Everything Clicked**
After weeks of iteration, Miguel finally reached **the moment every ML engineer dreams of**:

🎉 **The first fully automated inference with a beautifully annotated output image, complete with a project-branded frame and QR code linking to the repository.**

It wasn’t just a functional win; it was a visual statement. **The project now had an identity.**

📸 *“I embedded the inference metadata and repo link inside the EXIF UserComment tag—an Easter egg to show my interest in data integrity and tracking.”*

### **More Than Just a Model: The Power of Presentation**
Through this project, Miguel learned that **technical execution is only half the battle.** The other half? **Storytelling.**

💡 *“A data scientist’s job isn’t just to build models—it’s to communicate findings in a way that excites stakeholders.”*

This realization led him to explore different ways of showcasing his work. The result?
✅ **A rich CLI experience for easy experimentation**
✅ **Fully documented code for reproducibility**
✅ **Publicly accessible demos to engage potential employers**

### **A Call to Action: Let’s Talk!**
Miguel is now actively seeking his **first opportunity in the IT sector.** He’s proven that he can:
✔️ **Design and execute a full ML pipeline from scratch**
✔️ **Combine deep learning with algorithmic ingenuity**
✔️ **Present technical solutions in a clear and engaging way**
✔️ **Build user-friendly tools that enhance accessibility**

💬 **If you’re looking for someone who merges strong technical execution, structured problem-solving, and creative thinking in AI, let’s connect!**

📩 **Contact:** [Miguel’s LinkedIn](https://www.linkedin.com/in/MiguelDiLalla)  
🔗 **Project Repo:** [GitHub](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)

---

# **🚀 Development Stages & Key Decisions**

## **🔷 Stage 1: The Transition to Tech**
**(From Hospitality to Machine Learning)**  
- **Realization**: Hospitality was not aligned with long-term goals. Needed an intellectually fulfilling career.  
- **First Steps**: Google Data Analytics course → Bootcamp (Hack a Boss) → Hands-on ML learning with ChatGPT & GitHub Copilot.  
- **Project Ideation**: Inspired by childhood LEGO experience → Formulating the AI challenge: *“Can I teach a machine to do what I do instinctively?”*

---

## **🔷 Stage 2: Initial Project Scope & Assumptions**  
**(Early Optimism & First Roadblocks)**  
- **Initial Goal**: Train a single deep learning model to output brick dimensions (direct numerical classification).  
- **Dataset Planning**:
  ✅ Identified 27 classes of LEGO bricks.
  ✅ Gathered all available colors & materials for dataset diversity.
  ✅ Captured thousands of images under varying lighting conditions & angles.  
- **First Major Challenge**: Custom ML models **failed to generalize** brick dimensions reliably. Needed a rethink.  

---

## **🔷 Stage 3: Strategic Pivot**  
**(Breaking the Problem into Solvable Parts)**  
- **Key Insight**: Predicting numerical dimensions directly was too complex → **Divide the task into smaller components**.  
- **New Modular Approach:**
  1️⃣ **Detect LEGO bricks in the image** (Bounding Box Detection)  
  2️⃣ **Crop and isolate detected bricks**  
  3️⃣ **Detect studs within the cropped images** (Keypoint Detection)  
  4️⃣ **Use geometry-based logic to infer dimensions**  
- **Technology Evaluation:**  
  - ❌ **Google Mediapipe** → Limited control.  
  - ❌ **Detectron** → Overkill for this application.  
  - ✅ **YOLOv8** → Best tradeoff between usability, speed & accuracy.  

---

## **🔷 Stage 4: Optimizing Detection & Classification**  
**(Combining ML & Traditional Algorithms)**  
- **Stud Detection Strategy:**
  ✅ Trained a second YOLO model for keypoint detection of studs.  
  ✅ Annotated hundreds of images manually.  
  ✅ Converted keypoints → bounding boxes using a custom algorithm.  
- **Final Classification Logic:**
  - Used detected studs as **Cartesian plane points**.  
  - Applied **linear regression** to determine row/column alignment.  
  - Differentiated bricks with the same number of studs (e.g., 2x4 vs. 1x8).  

---

## **🔷 Stage 5: Infrastructure & Usability**  
**(Making the Project Accessible & Scalable)**  
- **CLI Development:**
  ✅ Designed a fully documented **Rich-based CLI** for seamless execution.  
  ✅ Integrated **emoji-based UX** for better visual clarity.  
  ✅ Simplified execution: One command to reproduce all steps.  
- **Metadata Fingerprinting:**
  ✅ Embedded inference metadata & repo link inside EXIF UserComment tags.  
  ✅ Showcases awareness of **data integrity & tracking** (like a ‘cookie’ for images).  
- **Deployment Strategy Considerations:**
  - ❌ Initially explored **Docker**, but **dropped** due to portability concerns.  
  - ✅ **Pivoted to Streamlit & Kaggle Notebooks** for interactive demos.  

---

## **🔷 Stage 6: The Breakthrough Moment**  
**(When Everything Clicked)**  
- First fully processed image 🎉 → **Perfect annotations, branded frame & QR code linking to repo.**  
- Realization: **Presentation is as important as execution** → Data scientists need to communicate results effectively.  
- Decision to focus on creating **engaging, easy-to-access demos for potential employers.**  

---

## **🔷 Conclusion & Next Steps**  
- **Project Completed & Fully Documented** ✅  
- **Now Seeking Opportunities** → First IT job leveraging problem-solving, ML, and UX skills.  
- **Public Demos & Portfolio Expansion** → Making results accessible for hiring managers & tech audiences.  

📩 **Let’s Connect!**  
🔗 **[GitHub Repo](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)**  
🔗 **[LinkedIn Profile](https://www.linkedin.com/in/MiguelDiLalla)**  


---

# **Technical Proficiencies**

Below is a concise overview of the **key technical skill sets** demonstrated throughout the development of the LEGO Bricks ML Vision project.

---

## **1. Machine Learning & Deep Learning**

- **YOLOv8 (Ultralytics)**: Trained single-class and multiclass models; configured advanced hyperparameters (epochs, batch size, early stopping).
- **PyTorch**: Managed GPU/CPU devices, loaded and fine-tuned pretrained models (`yolov8n.pt`).
- **Hybrid ML Approaches**:
  - Combined deep learning detection with classical geometry-based algorithms for stud layout classification.
  - Deployed keypoint detection (for studs) and bounding box detection (for bricks).
- **Regression Analysis**: Used linear regression + threshold logic to differentiate similar bricks (e.g., 2x4 vs. 1x8) based on stud positions.

---

## **2. Data Handling & Pipeline Automation**

- **Annotation & Format Conversion**:
  - Employed LabelMe for manual annotations, then scripted JSON → YOLO .txt conversion.
  - Developed additional logic to handle keypoints → bounding boxes.
- **Data Augmentation & Splitting**:
  - Used Albumentations for random flips, brightness, and color jitter.
  - Programmatically created train/val/test splits with user-defined ratios.
- **EXIF Metadata Integration**:
  - Inserted inference data and repo links in `UserComment` tags.
  - Demonstrated advanced data stewardship and analytics potential.
- **Automated Scripts** (`train_singleclass.py`, `train_multiclass.py`):
  - End-to-end processes for dataset extraction, YOLO configuration, and archiving.

---

## **3. Software Engineering & Tooling**

- **CLI Development (Rich + Click)**:
  - Created a visually engaging command-line interface (`lego_cli.py`) for detection, inference, and data utilities.
  - Integrated progress bars, emojis, and color-coded logs.
- **Logging & Error Handling**:
  - Detailed logs with timestamps, graceful fallback if GPU is absent.
  - Clean, user-friendly error messages guiding next steps.
- **Modular Design**:
  - Organized code into `utils/` for data processing, detection, metadata, batch operations.
  - Clear function docstrings for maintainability.

---

## **4. System & Infrastructure Management**

- **Hardware Detection & Setup**:
  - Automated GPU detection, fallback to CPU with user confirmation.
  - Supported multiple GPUs by building device strings (e.g., `0,1`).
- **Directory & Artifact Management**:
  - Cache directories for extracted datasets, ephemeral logs, final models.
  - Timestamped results folders; zip archiving for portability.
- **Docker vs Local Execution**:
  - Explored containerization but pivoted to direct GitHub-based solutions for simpler distribution.

---

## **5. Integration & Code Structure**

- **Repository Architecture**:
  - `presentation/` for compressed datasets & pretrained models.
  - `cache/` & `results/` for ephemeral and final outputs.
  - `scripts/` & `utils/` for modular code (data_utils, detection_utils, exif_utils, etc.).
- **Config-Driven**:
  - YAML-based dataset configurations for single-class vs. multiclass setups.
  - Streamlined adaptation to new classes or label formats.

---

## **6. Testing & Validation**

- **Unit Testing**:
  - Basic checks ensuring JSON → YOLO transformation correctness.
- **Manual Evaluation**:
  - Verified bounding boxes, stud detection, and classification results under varied lighting and angles.
- **Iterative Debugging**:
  - Identified confusion cases (similar bricks), introduced regression-based logic.

---

## **7. Communication & Collaboration**

- **Git & GitHub**:
  - Clear commit messaging, branching, and merging strategies.
  - Shared code for open collaboration and issue tracking.
- **Rich Documentation**:
  - Markdown-based instructions, docstrings, and `--help` references.
  - Emphasis on guiding colleagues and potential end users.

---

## **8. Overall Proficiency Summary**

**Miguel’s contributions** to the LEGO Bricks ML Vision project highlight a **well-rounded technical profile** encompassing:
- **Advanced ML Techniques & Fine-Tuning**: YOLO-based pipelines, keypoint detection, hybrid approaches.
- **Robust Data Workflow**: Annotation, augmentation, EXIF metadata usage, automated splitting.
- **Software Engineering Best Practices**: Structured CLI, clear folder organization, strong logging.
- **Infrastructure & Deployment Awareness**: GPU/CPU detection, caching, portability considerations.
- **Team Collaboration & Documentation**: Rich visuals, thorough docstrings, streamlined user experience.

**Together**, these proficiencies enable **both creative and production-ready** ML solutions—bridging the gap between experimental prototypes and reliable, user-friendly software.

---

For a deeper look into each script and methodology, explore:  
🔗 **[LEGO Bricks ML Vision Repo](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)**


