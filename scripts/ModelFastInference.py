#!/usr/bin/env python
"""
YOLO Model Testing UI
A minimalist dark-themed interface for testing YOLO models with drag-and-drop support.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Rest of imports
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QStatusBar, QScrollArea)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QPalette, QColor, QDropEvent, QDragEnterEvent, QImage
import cv2
import numpy as np
from ultralytics import YOLO
import json

# Project imports
from utils.detection_utils import detect_bricks, detect_studs
from utils.pipeline_utils import run_full_algorithm
from utils.metadata_utils import extract_metadata_from_yolo_result, format_detection_summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants for styling
DARK_GREY = "#2B2B2B"
DARKER_GREY = "#1E1E1E"
STATUS_BAR_HEIGHT = 60
MAX_LOG_LINES = 3

class DragDropLabel(QLabel):
    """Custom QLabel with drag and drop support."""
    
    def __init__(self, parent=None, accept_types=None):
        super().__init__(parent)
        self.main_window = parent  # Store reference to main window
        self.accept_types = accept_types or [".jpg", ".jpeg", ".pt"]
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {DARKER_GREY};
                border: 2px dashed #555555;
                border-radius: 5px;
                color: white;
                padding: 10px;
            }}
        """)
        self.setAcceptDrops(True)
        self.setText("Drag & Drop\nFiles Here")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if all(Path(url.toLocalFile()).suffix.lower() in self.accept_types 
                  for url in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        # Find the YOLOTestingUI parent
        parent = self
        while parent and not isinstance(parent, YOLOTestingUI):
            parent = parent.parent()
        if parent:
            parent.process_dropped_files(files)

class MetadataPanel(QWidget):
    """Panel for displaying metadata in a monospace font with dark background."""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        
        # Content in ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {DARKER_GREY};
                border-radius: 5px;
                border: none;
            }}
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        self.content = QLabel()
        self.content.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                color: #50fa7b;  /* Green text color */
                font-family: Consolas, monospace;
                padding: 10px;
            }
        """)
        self.content.setWordWrap(True)
        self.content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        content_layout.addWidget(self.content)
        scroll.setWidget(content_widget)
        
        layout.addWidget(title_label)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def update_content(self, text):
        self.content.setText(text)

class StatusBar(QStatusBar):
    """Custom status bar that shows the last 3 log messages."""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            QStatusBar {{
                background-color: {DARKER_GREY};
                color: white;
                padding: 5px;
            }}
        """)
        self.log_messages = []
        self.setFixedHeight(STATUS_BAR_HEIGHT)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        
        # Keep only the last 3 messages
        if len(self.log_messages) > MAX_LOG_LINES:
            self.log_messages.pop(0)
            
        self.showMessage("\n".join(self.log_messages))

class YOLOTestingUI(QMainWindow):
    """Main window for the YOLO Model Testing UI."""
    
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.current_image = None
        self.setup_ui()
        self.setAcceptDrops(True)  # Enable drops for the main window
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for the main window."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            valid_extensions = ['.pt', '.jpg', '.jpeg']
            if all(Path(url.toLocalFile()).suffix.lower() in valid_extensions 
                  for url in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop events for the main window."""
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self.process_dropped_files(files)

    def setup_ui(self):
        self.setWindowTitle("YOLO Model Testing")
        self.setStyleSheet(f"background-color: {DARK_GREY};")
        self.setMinimumSize(1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout for central widget
        main_layout = QVBoxLayout(central_widget)
        
        # Top row with image panels
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Original image panel
        self.original_image = DragDropLabel(self, accept_types=[".jpg", ".jpeg", ".pt"])
        self.original_image.setMinimumSize(QSize(400, 400))
        
        # Annotated image panel
        self.annotated_image = QLabel()
        self.annotated_image.setStyleSheet(f"""
            QLabel {{
                background-color: {DARKER_GREY};
                color: white;
                border-radius: 5px;
            }}
        """)
        self.annotated_image.setAlignment(Qt.AlignCenter)
        self.annotated_image.setMinimumSize(QSize(400, 400))
        self.annotated_image.setText("Processed Image\nWill Appear Here")
        
        top_layout.addWidget(self.original_image)
        top_layout.addWidget(self.annotated_image)
        
        # Bottom row with metadata panels
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Model info panel
        self.model_info = MetadataPanel("Model Information", self)
        self.model_info.update_content("Drag & drop a .pt file to load model")
        
        # Image metadata panel
        self.image_metadata = MetadataPanel("Image Metadata", self)
        self.image_metadata.update_content("Drag & drop an image to view metadata")
        
        bottom_layout.addWidget(self.model_info)
        bottom_layout.addWidget(self.image_metadata)
        
        # Add widgets to main layout
        main_layout.addWidget(top_widget)
        main_layout.addWidget(bottom_widget)
        
        # Status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.log("Ready to process files")

    def process_dropped_files(self, files):
        """Process dropped files based on their type."""
        for file_path in files:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() in ['.pt']:
                self.load_model(file_path)
            elif file_path.suffix.lower() in ['.jpg', '.jpeg']:
                self.load_image(file_path)
            else:
                self.status_bar.log(f"Unsupported file type: {file_path.suffix}")

    def load_model(self, model_path):
        """Load and validate a YOLO model."""
        try:
            self.status_bar.log(f"Loading model: {model_path.name}")
            
            # Clear existing model and reset annotated image
            if self.current_model is not None:
                del self.current_model
            self.current_model = None
            self.annotated_image.setText("Processed Image\nWill Appear Here")
            
            # Load new model
            model = YOLO(str(model_path))
            
            # Extract model information
            num_classes = len(model.names)
            class_dict = model.names
            
            # Update model info panel
            info_text = (f"Model: {model_path.name}\n"
                        f"Number of classes: {num_classes}\n"
                        f"Classes:\n")
            for idx, name in class_dict.items():
                info_text += f"  {idx}: {name}\n"
            
            self.model_info.update_content(info_text)
            self.current_model = model
            self.status_bar.log("Model loaded successfully")
            
            # Reprocess current image if one is loaded
            if self.current_image is not None:
                self.process_image(self.current_image)
            
        except Exception as e:
            self.model_info.update_content(f"Error loading model:\n{str(e)}")
            self.status_bar.log("Failed to load model")

    def load_image(self, image_path):
        """Load and process an image."""
        try:
            # Read and validate image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Invalid or corrupted image file")
            
            # Store current image
            self.current_image = img.copy()
            
            # Update original image display
            height, width = img.shape[:2]
            self.display_image(self.original_image, img)
            
            # Update image metadata
            metadata_text = (f"Filename: {image_path.name}\n"
                           f"Dimensions: {width}x{height}\n"
                           f"Size: {os.path.getsize(image_path) / 1024:.1f} KB")
            self.image_metadata.update_content(metadata_text)
            
            # Process with model if available
            if self.current_model:
                self.process_image(img)
            else:
                self.status_bar.log("No model loaded - load a .pt file first")
                
        except Exception as e:
            self.image_metadata.update_content(f"Error loading image:\n{str(e)}")
            self.status_bar.log("Failed to load image")

    def process_image(self, img):
        """Process an image with the current model."""
        try:
            self.status_bar.log("Processing image with model...")
            results = self.current_model(img)
            
            # Extract and display metadata
            metadata = extract_metadata_from_yolo_result(results, img)
            metadata_text = json.dumps(metadata, indent=2)
            self.image_metadata.update_content(metadata_text)
            
            # Display annotated image
            annotated_img = results[0].plot()
            self.display_image(self.annotated_image, annotated_img)
            self.status_bar.log("Image processed successfully")
        except Exception as e:
            self.status_bar.log(f"Error processing image: {str(e)}")
            self.image_metadata.update_content(f"Error processing image:\n{str(e)}")

    def display_image(self, label, img):
        """Display an image in a label while maintaining aspect ratio."""
        if isinstance(img, np.ndarray):
            height, width = img.shape[:2]
            label_size = label.size()
            
            # Calculate scaling factor to fit in label
            scale = min(label_size.width() / width,
                       label_size.height() / height)
            
            # Resize image
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(img, (new_width, new_height))
            
            # Convert to QPixmap and display
            if len(img.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            qimg = QPixmap.fromImage(QImage(resized.data, new_width, new_height,
                                          resized.strides[0], QImage.Format_RGB888))
            label.setPixmap(qimg)

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(DARK_GREY))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(DARKER_GREY))
    palette.setColor(QPalette.AlternateBase, QColor(DARK_GREY))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(DARK_GREY))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor("#3377b0"))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = YOLOTestingUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()