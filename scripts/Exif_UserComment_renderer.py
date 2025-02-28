'''
EXIF UserComment Renderer

This script creates a GUI window that allows users to drag and drop image files.
When a valid image is dropped, it displays the image as a miniature and shows
the contents of the EXIF UserComment tag below it.

Usage:
- Run the script
- Drag and drop an image file onto the window
- View the image and its EXIF UserComment metadata
- Drag another image to update the display
- Close the window to exit
'''

import os
import sys
import json
import tkinter as tk
from tkinter import scrolledtext, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk
import piexif

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from scripts.Legacy_scripts.model_utils import read_exif
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False
    print("Warning: Couldn't import read_exif from utils.model_utils")
    print("Will use local implementation instead")

    # Local implementation of read_exif if import fails
    def read_exif(image_path, TREE=None):
        """Local implementation of read_exif for standalone use"""
        if TREE is None:
            TREE = {
                "boxes_coordinates": {},
                "orig_shape": [0, 0],
                "speed": {
                    "preprocess": 0.0,
                    "inference": 0.0,
                    "postprocess": 0.0
                },
                "mode": "",
                "path": "",
                "os_full_version_name": "",
                "processor": "",
                "architecture": "",
                "hostname": "",
                "timestamp": "",
                "annotated_image_path": "",
                "json_results_path": "",
                "TimesScanned": 0,
                "Repository": "",
                "message": ""
            }
        
        try:
            with Image.open(image_path) as image:
                exif_bytes = image.info.get("exif")
                if not exif_bytes:
                    print(f"⚠️ No EXIF data found in {image_path}")
                    return {}

                exif_dict = piexif.load(exif_bytes)
        except Exception as e:
            print(f"❌ Failed to open image {image_path} > {e}")
            return {}

        user_comment_tag = piexif.ExifIFD.UserComment
        user_comment = exif_dict.get("Exif", {}).get(user_comment_tag, b"")
        if not user_comment:
            print(f"⚠️ No UserComment tag found in {image_path}")
            return {}

        try:
            comment_str = user_comment.decode('utf-8', errors='ignore')
            metadata = json.loads(comment_str)
            # Ensure defaults from TREE are present
            for key, default in TREE.items():
                metadata.setdefault(key, default)
            times = metadata.get("TimesScanned", 0)
            if times:
                print(f"🔄 Image {image_path} has been scanned {times} time(s)")
            else:
                print(f"🆕 Image {image_path} has not been scanned before")
            return metadata
        except Exception as e:
            print(f"❌ Failed to parse EXIF metadata from {image_path}: {e}")
            return {}


class ExifUserCommentRenderer:
    def __init__(self, root):
        self.root = root
        self.root.title("EXIF UserComment Renderer")
        self.root.geometry("800x600")
        self.root.configure(bg='#2d2d2d')
        
        # Configure the main window
        self.setup_ui()
        self.setup_drop_target()
        
        # Initialize variables
        self.current_image_path = None
        self.photo_image = None  # Keep reference to prevent garbage collection
        
    def setup_ui(self):
        # Create header
        header_frame = tk.Frame(self.root, bg="#1a1a1a")
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        title = tk.Label(header_frame, text="EXIF UserComment Renderer", 
                         font=title_font, bg="#1a1a1a", fg="white")
        title.pack(pady=5)
        
        # Create main frame with image area and metadata area
        main_frame = tk.Frame(self.root, bg='#2d2d2d')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Image area
        self.image_frame = tk.Frame(main_frame, bg='#2d2d2d')
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.image_label = tk.Label(self.image_frame, bg='#3d3d3d', 
                                   text="Drag and drop an image file here", 
                                   fg="white", height=10)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Metadata area with scrolled text
        metadata_frame = tk.Frame(main_frame, bg='#2d2d2d')
        metadata_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        metadata_label = tk.Label(metadata_frame, text="EXIF UserComment Data:", 
                                 bg='#2d2d2d', fg="white", anchor="w")
        metadata_label.pack(fill=tk.X)
        
        self.metadata_text = scrolledtext.ScrolledText(metadata_frame, bg='black', 
                                                      fg='white', height=15,
                                                      font=("Consolas", 10))
        self.metadata_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready - Drop an image file", 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                                  bg="#1a1a1a", fg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_drop_target(self):
        # Enable drag and drop for the window
        self.root.drop_target_register("DND_Files")
        self.root.dnd_bind('<<Drop>>', self.handle_drop)
    
    def handle_drop(self, event):
        # Get the dropped file path
        file_path = event.data
        
        # Remove quotes if present (happens in some systems)
        if file_path.startswith('{') and file_path.endswith('}'): 
            file_path = file_path[1:-1]
        if file_path.startswith('"') and file_path.endswith('"'): 
            file_path = file_path[1:-1]
            
        # Check if it's an image file
        if not self.is_valid_image(file_path):
            messagebox.showerror("Error", "Please drop a valid image file.")
            self.status_bar.config(text=f"Error: Not a valid image file")
            return
            
        # Process and display the image
        self.process_image(file_path)
    
    def is_valid_image(self, file_path):
        """Check if the file is a valid image"""
        try:
            img = Image.open(file_path)
            img.verify()
            return True
        except Exception:
            return False
    
    def process_image(self, file_path):
        """Process the dropped image and display it with its EXIF data"""
        self.status_bar.config(text=f"Processing: {os.path.basename(file_path)}")
        self.root.update()
        
        # Save the current path
        self.current_image_path = file_path
        
        # Display the image
        self.display_image(file_path)
        
        # Extract and display EXIF data
        self.extract_and_display_exif(file_path)
        
        self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")
    
    def display_image(self, file_path):
        """Display the image in the image area"""
        try:
            # Open and resize the image while maintaining aspect ratio
            img = Image.open(file_path)
            img.thumbnail((400, 300))
            self.photo_image = ImageTk.PhotoImage(img)
            
            # Update the image label
            self.image_label.config(image=self.photo_image, text="")
            self.image_label.image = self.photo_image  # Keep a reference
            
        except Exception as e:
            self.image_label.config(text=f"Error displaying image: {e}", image="")
    
    def extract_and_display_exif(self, file_path):
        """Extract EXIF data and display it"""
        try:
            # Clear previous content
            self.metadata_text.delete(1.0, tk.END)
            
            # Get EXIF data using read_exif function
            metadata = read_exif(file_path)
            
            if not metadata:
                self.metadata_text.insert(tk.END, "No EXIF UserComment data found in this image.")
                return
                
            # Format the metadata as pretty JSON
            formatted_json = json.dumps(metadata, indent=4)
            self.metadata_text.insert(tk.END, formatted_json)
            
        except Exception as e:
            self.metadata_text.delete(1.0, tk.END)
            self.metadata_text.insert(tk.END, f"Error extracting EXIF data: {e}")


def main():
    # Check if TkinterDnD is available
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        messagebox.showerror(
            "Missing Dependency", 
            "This application requires tkinterdnd2.\n"
            "Please install it using:\n"
            "pip install tkinterdnd2"
        )
        print("Error: tkinterdnd2 is required. Install using: pip install tkinterdnd2")
        # Fall back to regular Tk without drag & drop
        root = tk.Tk()
        root.withdraw()
        sys.exit(1)
    
    app = ExifUserCommentRenderer(root)
    root.mainloop()


if __name__ == "__main__":
    main()


