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
import subprocess

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import exif_utils

# Add modern theme constants
THEME = {
    'bg_dark': '#1E1E1E',
    'bg_medium': '#252526',
    'bg_light': '#2D2D2D',
    'text': '#FFFFFF',
    'accent': '#007ACC',
    'border': '#3E3E42',
    'padding': 12,
    'button_disabled': '#4D4D4D'
}

class ExifUserCommentRenderer:
    def __init__(self, root):
        self.root = root
        self.root.title("EXIF Viewer")
        self.root.geometry("1200x700")  # Increased width to accommodate side-by-side layout
        self.root.configure(bg=THEME['bg_dark'])
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        self.setup_ui()
        self.setup_drop_target()
        self.current_image_path = None
        self.photo_image = None
        
    def setup_ui(self):
        # Header with minimal design
        header_frame = tk.Frame(self.root, bg=THEME['bg_dark'])
        header_frame.grid(row=0, column=0, sticky='ew', pady=(THEME['padding'], 0))
        header_frame.grid_columnconfigure(1, weight=1)  # Make the space between title and button flexible
        
        title_font = tkFont.Font(family="Segoe UI", size=14, weight="normal")
        title = tk.Label(header_frame, text="EXIF Metadata Viewer", 
                        font=title_font, bg=THEME['bg_dark'], 
                        fg=THEME['text'], pady=THEME['padding'])
        title.grid(row=0, column=0, sticky='w')
        
        # Add clean metadata button
        self.clean_button = tk.Button(
            header_frame,
            text="Clean Metadata",
            command=self.clean_metadata,
            bg=THEME['bg_medium'],
            fg=THEME['text'],
            activebackground=THEME['bg_light'],
            activeforeground=THEME['text'],
            relief=tk.FLAT,
            state='disabled'  # Initially disabled
        )
        self.clean_button.grid(row=0, column=2, sticky='e', padx=THEME['padding'])
        
        # Main container frame
        container = tk.Frame(self.root, bg=THEME['bg_dark'])
        container.grid(row=1, column=0, sticky='nsew', padx=THEME['padding'], 
                      pady=THEME['padding'])
        container.grid_columnconfigure(0, weight=1)  # Image panel
        container.grid_columnconfigure(1, weight=1)  # Metadata panel
        container.grid_rowconfigure(0, weight=1)
        
        # Image panel
        self.image_frame = tk.Frame(container, bg=THEME['bg_medium'],
                                  highlightbackground=THEME['border'],
                                  highlightthickness=1)
        self.image_frame.grid(row=0, column=0, sticky='nsew', 
                            padx=(0, THEME['padding']/2))
        
        self.image_label = tk.Label(self.image_frame, bg=THEME['bg_medium'],
                                  text="Drop image here",
                                  fg=THEME['text'], height=8)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Metadata panel
        metadata_container = tk.Frame(container, bg=THEME['bg_medium'],
                                    highlightbackground=THEME['border'],
                                    highlightthickness=1)
        metadata_container.grid(row=0, column=1, sticky='nsew',
                              padx=(THEME['padding']/2, 0))
        
        # Use a modern monospace font
        self.metadata_text = scrolledtext.ScrolledText(
            metadata_container,
            bg=THEME['bg_light'],
            fg=THEME['text'],
            font=("Cascadia Code", 10),
            wrap=tk.NONE,
            borderwidth=0,
            highlightthickness=0,
            padx=THEME['padding'],
            pady=THEME['padding']
        )
        self.metadata_text.pack(fill=tk.BOTH, expand=True)
        
        # Minimal status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bg=THEME['bg_medium'],
            fg=THEME['text'],
            anchor=tk.W,
            padx=THEME['padding'],
            pady=6
        )
        self.status_bar.grid(row=2, column=0, sticky='ew')
        
        # Add hover effects
        self.add_hover_effects()
    
    def add_hover_effects(self):
        def on_enter(event):
            self.image_label.config(bg=THEME['bg_light'])
        
        def on_leave(event):
            self.image_label.config(bg=THEME['bg_medium'])
        
        self.image_label.bind('<Enter>', on_enter)
        self.image_label.bind('<Leave>', on_leave)
    
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
        self.clean_button.config(state='normal')  # Enable the clean button
    
    def display_image(self, file_path):
        """Display the image in the image area"""
        try:
            img = Image.open(file_path)
            # Calculate aspect ratio for better fit in the side panel
            display_height = 600  # Maximum height
            display_width = 500   # Maximum width
            
            # Calculate new dimensions maintaining aspect ratio
            img_ratio = img.width / img.height
            if img_ratio > display_width/display_height:
                new_width = display_width
                new_height = int(display_width / img_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * img_ratio)
                
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.photo_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo_image, text="")
            self.image_label.image = self.photo_image
            
        except Exception as e:
            self.image_label.config(
                text=f"Error displaying image",
                image="",
                fg=THEME['accent']
            )
    
    def extract_and_display_exif(self, file_path):
        """Extract EXIF data and display it"""
        try:
            # Clear previous content
            self.metadata_text.delete(1.0, tk.END)
            
            # Get EXIF data using exif_utils
            metadata = exif_utils.read_exif(file_path)
            
            if not metadata:
                self.metadata_text.insert(tk.END, "No EXIF UserComment data found in this image.")
                return
                
            # Format the metadata as pretty JSON
            formatted_json = json.dumps(metadata, indent=4)
            self.metadata_text.insert(tk.END, formatted_json)
            
        except Exception as e:
            self.metadata_text.delete(1.0, tk.END)
            self.metadata_text.insert(tk.END, f"Error extracting EXIF data: {e}")

    def clean_metadata(self):
        """Clean the metadata of the current image"""
        if not self.current_image_path:
            return
        
        try:
            # Get the parent folder of the current image
            parent_folder = os.path.dirname(self.current_image_path)
            
            # Construct and execute the CLI command
            cli_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lego_cli.py')
            
            # Create a process that we can interact with, with UTF-8 encoding
            startupinfo = None
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            process = subprocess.Popen(
                ['python', cli_path, 'metadata', 'clean-batch', parent_folder],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                startupinfo=startupinfo
            )
            
            # Send 'Y' to the process
            stdout, stderr = process.communicate('Y\n')
            
            if process.returncode == 0:
                messagebox.showinfo("Success", "Metadata cleaned successfully.")
                self.status_bar.config(text="Metadata cleaned successfully.")
                
                # Refresh the display
                self.process_image(self.current_image_path)
            else:
                messagebox.showerror("Error", f"Failed to clean metadata: {stderr}")
                self.status_bar.config(text="Error: Failed to clean metadata")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clean metadata: {str(e)}")
            self.status_bar.config(text="Error: Failed to clean metadata")

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


