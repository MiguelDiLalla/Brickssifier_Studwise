#!/usr/bin/env python
"""
LEGO Bricks ML Vision - Fast Inference UI
A minimalist red-themed interface for LEGO brick detection with drag-and-drop support.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to Python path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QLabel, QScrollArea, QHBoxLayout
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent

# Import project utilities
from utils.pipeline_utils import run_full_algorithm
from utils.config_utils import setup_utils

# Constants for styling
DARK_RED = "#2B0000"
DARKER_RED = "#1E0000"
ACCENT_RED = "#FF3333"
MIN_WIDTH = 800   # Back to original size
MIN_HEIGHT = 600  # Back to original size

class DragDropLabel(QLabel):
    """Custom QLabel with drag and drop support for images."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {DARKER_RED};
                border: 2px dashed {ACCENT_RED};
                border-radius: 5px;
                color: white;
                padding: 10px;
            }}
        """)
        self.setAcceptDrops(True)
        self.setText("Drag & Drop\nImage Files Here")
        self.setMinimumSize(MIN_WIDTH // 2 - 20, MIN_HEIGHT - 20)  # Adjust for padding

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if all(url.toLocalFile().lower().endswith(('.jpg', '.jpeg', '.png')) 
                  for url in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        parent = self
        while parent and not isinstance(parent, BrickDetectionUI):
            parent = parent.parent()
        if parent:
            parent.process_image(files[0])  # Process first image only

class MetadataPanel(QWidget):
    """Panel for displaying inference metadata."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {DARKER_RED};
                border-radius: 5px;
                border: none;
            }}
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        self.content = QLabel()
        self.content.setStyleSheet("""
            QLabel {
                background-color: #1E0000;
                color: #FF6666;
                font-family: Consolas, monospace;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.content.setWordWrap(True)
        self.content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        content_layout.addWidget(self.content)
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def update_content(self, text):
        self.content.setText(text)

class BrickDetectionUI(QMainWindow):
    """Main window for the LEGO Brick Detection UI."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_output_dir = None
        
    def setup_ui(self):
        self.setWindowTitle("LEGO Brick Detection")
        self.setStyleSheet(f"background-color: {DARK_RED};")
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)  # Add some space between panels
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add margins around the edges

        # Image display panel
        self.image_panel = DragDropLabel(self)
        self.image_panel.mousePressEvent = self.open_output_folder
        self.image_panel.setMinimumWidth((MIN_WIDTH // 2) - 20)

        # Metadata panel
        self.metadata_panel = MetadataPanel(self)
        self.metadata_panel.setMinimumWidth((MIN_WIDTH // 2) - 20)
        self.metadata_panel.update_content("Drag & drop an image to analyze")
        
        main_layout.addWidget(self.image_panel)
        main_layout.addWidget(self.metadata_panel)
        
        # Set stretch factors to make panels equal width
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        
    def process_image(self, image_path):
        """Process an image using the full detection pipeline."""
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(PROJECT_ROOT, "cache", f"inference_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run full detection pipeline
            result = run_full_algorithm(
                image_path,
                save_annotated=True,
                output_folder=output_dir,
                force_rerun=True
            )
            
            if result is None:
                raise Exception("Detection failed")
            
            # Find and display composite analysis image
            analysis_image = os.path.join(output_dir, "full_analysis.jpg")
            metadata_file = os.path.join(output_dir, "full_analysis_metadata.json")
            
            if os.path.exists(analysis_image):
                self.current_output_dir = output_dir
                self.display_image(analysis_image)
                
                # Display metadata from file if available
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            self.metadata_panel.update_content(self.format_metadata(metadata))
                    except Exception as e:
                        self.metadata_panel.update_content(f"Error loading metadata: {str(e)}")
                else:
                    self.metadata_panel.update_content("Metadata file not found")
            else:
                self.metadata_panel.update_content("Error: Analysis image not found")
                
        except Exception as e:
            self.metadata_panel.update_content(f"Error: {str(e)}")

    def format_metadata(self, metadata):
        """Format metadata from JSON file."""
        formatted_text = []
        
        # Get brick detection info
        brick_info = metadata.get("brick_detection", {})
        brick_count = brick_info.get("count", 0)
        formatted_text.append(f"Total bricks detected: {brick_count}")
        
        # Display individual stud detection results
        stud_detections = metadata.get("stud_detection", [])
        if stud_detections:
            formatted_text.append("\nDetailed Analysis:")
            
            for detection in stud_detections:
                idx = detection.get("index", 0)
                dim = detection.get("dimension", "Unknown")
                timestamp = detection.get("timestamp", "").split("T")[0]  # Just get the date part
                
                formatted_text.append(f"\nBrick {idx + 1}:")
                formatted_text.append(f"- Dimension: {dim}")
                if timestamp:
                    formatted_text.append(f"- Analyzed: {timestamp}")
        
        # Add version info if available
        version = metadata.get("version", "")
        if version:
            formatted_text.append(f"\nAnalysis version: {version}")
                    
        return "\n".join(formatted_text)

    def display_image(self, image_path):
        """Display an image in the image panel while maintaining aspect ratio."""
        pixmap = QPixmap(image_path)
        scaled = pixmap.scaled(
            self.image_panel.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_panel.setPixmap(scaled)

    def open_output_folder(self, event):
        """Open the output folder when image panel is clicked."""
        if self.current_output_dir and os.path.exists(self.current_output_dir):
            os.startfile(self.current_output_dir)

    def closeEvent(self, event):
        """Clean up cache folder when closing."""
        try:
            import shutil
            cache_dir = os.path.join(PROJECT_ROOT, "cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir)
        finally:
            event.accept()

def main():
    app = QApplication(sys.argv)
    window = BrickDetectionUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()