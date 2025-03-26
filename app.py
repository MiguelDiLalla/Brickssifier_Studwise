import streamlit as st
import os
from PIL import Image
from pathlib import Path
import time
import sys
from io import BytesIO
import json
from streamlit_image_select import image_select
import base64

# Import the CLI interface
import sys
sys.path.append(".")
import lego_cli
from click.testing import CliRunner

# Page config
st.set_page_config(
    page_title="🧱 LEGO Bricks ML Vision Demo",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar content
st.sidebar.image("presentation/logo.png")
st.sidebar.title("👋 About")
st.sidebar.markdown("""
### 👨‍💻 Professional Profile
- 🔬 Aspiring Junior Data Scientist
- 🤖 Machine Learning & Computer Vision
- 🎯 Focus on Deep Learning Applications

### 🔗 Connect with Me
[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/MiguelDiLalla)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/MiguelDiLalla)

### 📚 Project Repository
[![View on GitHub](https://img.shields.io/badge/GitHub-View_Repository-blue?style=for-the-badge&logo=github)](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 About This Project
This demo showcases a computer vision system that:
1. 🔍 Detects LEGO bricks in images
2. 🎯 Identifies individual studs
3. 📏 Classifies brick dimensions
""")

# Function to load test images based on tab type
@st.cache_data
def load_test_images(tab_type):
    """Load test images based on tab type.
    
    Args:
        tab_type (str): Type of detection ('brick', 'stud', or 'dimension')
        
    Returns:
        list: Sorted list of image paths and their captions
    """
    if tab_type in ['brick', 'dimension']:
        test_images_path = Path("presentation/Test_images/BricksPics")
    else:  # stud detection
        test_images_path = Path("presentation/Test_images/StudsPics")
    
    images = []
    captions = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        found_paths = [str(p) for p in test_images_path.glob(ext)]
        for path in found_paths:
            try:
                # Keep track of both the path (for CLI) and loaded image (for display)
                img = Image.open(path)
                # Resize if too large while maintaining aspect ratio
                if img.width > 800 or img.height > 800:
                    ratio = min(800/img.width, 800/img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                images.append(path)  # Store path for CLI
                captions.append(Path(path).stem)
            except Exception as e:
                st.warning(f"Could not load image {path}: {str(e)}")
                continue
    
    # Sort by captions to maintain consistent order
    sorted_pairs = sorted(zip(images, captions), key=lambda x: x[1])
    images, captions = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    return list(images), list(captions)

# Initialize virtual folders in session state
if 'virtual_outputs' not in st.session_state:
    st.session_state['virtual_outputs'] = {
        'Brick Detection': {'images': [], 'metadata': []},
        'Stud Detection': {'images': [], 'metadata': []},
        'Dimension Classification': {'images': [], 'metadata': []},
        'Multiclass DEMO': {'images': [], 'metadata': []}  # Add new tab storage
    }

# Initialize uploaded images in session state
if 'uploaded_images' not in st.session_state:
    st.session_state['uploaded_images'] = {}

# Function to process single image through CLI
def process_single_image(image_path, tab_name):
    """Process an image using the appropriate CLI command based on tab name."""
    
    # Create a temporary output directory for CLI results
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        
        # Handle BytesIO or UploadedFile by saving to temporary file
        temp_input = None
        if isinstance(image_path, (BytesIO, st.runtime.uploaded_file_manager.UploadedFile)):
            # Create a consistent filename for uploaded images
            temp_input = os.path.join(temp_dir, "input_image.jpg")
            try:
                # Get image data regardless of input type
                if isinstance(image_path, st.runtime.uploaded_file_manager.UploadedFile):
                    file_key = f'uploaded_{tab_name}'
                    image_data = st.session_state['uploaded_images'][file_key]
                else:
                    image_data = image_path.read()
                
                # Write to temp file
                with open(temp_input, 'wb') as f:
                    f.write(image_data)
                image_path = temp_input
            except Exception as e:
                st.error(f"Failed to save temporary file: {str(e)}")
                return None, None

        # Create output directory structure that matches test images
        img_name = "input_image" if temp_input else Path(str(image_path)).stem
        img_output = os.path.join(temp_dir, img_name)
        os.makedirs(img_output, exist_ok=True)
        
        try:
            # Map tab names to CLI commands
            cli_command = ["detect-bricks"]  # Default command
            if tab_name.lower() == "stud detection":
                cli_command = ["detect-studs"]
            elif tab_name.lower() == "dimension classification":
                cli_command = ["infer"]
            elif tab_name.lower() == "multiclass demo":
                cli_command = ["detect-multiclass"]
            
            # Add common parameters
            cli_command.extend([
                "--image", str(image_path),
                "--save-annotated",
                "--output", img_output
            ])
            
            # Add --save-json for multiclass detection
            if tab_name.lower() == "multiclass demo":
                cli_command.append("--save-json")
            
            # Run CLI command
            start_time = time.time()
            result = runner.invoke(lego_cli.cli, cli_command)
            processing_time = time.time() - start_time
            
            if result.exit_code != 0:
                st.error(f"Processing failed: {result.output}")
                return None, None
            
            # Look for output image with consistent patterns
            output_path = None
            possible_patterns = [
                "cached_brick_detection.jpg",
                "cached_stud_detection.jpg",
                "brick_detection.jpg",
                "stud_detection.jpg",
                "annotated_image.jpg",
                "annotated.jpg",
                "full_analysis.jpg",
                "multiclass_detection.jpg"  # Add multiclass output pattern
            ]
            
            # Search in output directory and immediate subfolder
            for pattern in possible_patterns:
                # Try root output folder
                path = os.path.join(img_output, pattern)
                if os.path.exists(path):
                    output_path = path
                    break
                # Try in image name subfolder
                path = os.path.join(img_output, img_name, pattern)
                if os.path.exists(path):
                    output_path = path
                    break
                # Try in double-nested subfolder (for infer command)
                path = os.path.join(img_output, img_name, img_name, pattern)
                if os.path.exists(path):
                    output_path = path
                    break
            
            if not output_path or not os.path.exists(output_path):
                st.error("Failed to find output image")
                return None, None
            
            # Load and convert the output image
            result_img = Image.open(output_path)
            
            # Extract metadata from output
            metadata = None
            try:
                # Check for JSON metadata in output folder
                json_files = list(Path(img_output).glob('*.json'))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            metadata = json.load(f)
                        if isinstance(metadata, dict) and "boxes_coordinates" in metadata:
                            break
                    except json.JSONDecodeError:
                        continue

                if not metadata:
                    # If no JSON in output, check nested locations
                    possible_paths = [
                        os.path.join(img_output, "metadata.json"),
                        os.path.join(img_output, img_name, "metadata.json"),
                        os.path.join(img_output, img_name, img_name, "metadata.json"),
                        os.path.join(img_output, img_name, img_name, "full_analysis_metadata.json")  # Add infer command metadata pattern
                    ]
                    for json_path in possible_paths:
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r') as f:
                                    metadata = json.load(f)
                                if metadata:
                                    break
                            except json.JSONDecodeError:
                                continue

                    # If still no metadata, try to read from annotated image EXIF
                    if not metadata and output_path:
                        from utils.exif_utils import read_exif
                        metadata = read_exif(output_path) or {}
            
            except Exception as e:
                st.error(f"Error extracting metadata: {str(e)}")
                metadata = {}

            # Add processing time to metadata if not present
            if "processing_time" not in metadata:
                metadata.update({
                    "processing_time": f"{processing_time:.2f}s",
                    "command_used": " ".join(cli_command)
                })

            # Create formatted display version
            display_metadata = {}
            if metadata:
                # Only include relevant fields for display
                relevant_fields = [
                    "boxes_coordinates", "mode", "dimension",
                    "speed", "processing_time", "TimesScanned"
                ]
                display_metadata = {k: v for k, v in metadata.items() 
                                 if k in relevant_fields}

            # Store in virtual folder if we have results
            if result_img:
                # Convert PIL Image to bytes for storage
                img_byte_arr = BytesIO()
                result_img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Store in virtual folder
                st.session_state['virtual_outputs'][tab_name]['images'].append({
                    'name': f"{img_name}_result.jpg",
                    'data': img_byte_arr,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'metadata': metadata
                })
            
            return result_img, metadata

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None, None

# Function to clean test images metadata
def clean_test_images_metadata(tab_type):
    """Clean metadata from test images folder based on tab type."""
    with st.spinner('Preparing test images...'):
        if tab_type in ['brick', 'dimension']:
            test_folder = "presentation/Test_images/BricksPics"
        else:  # stud detection
            test_folder = "presentation/Test_images/StudsPics"
            
        # Use CLI runner to clean metadata with --force flag to skip confirmation
        runner = CliRunner(mix_stderr=False)  # Add mix_stderr=False
        result = runner.invoke(lego_cli.cli, ["metadata", "clean-batch", "--force", test_folder])
        if result.exit_code != 0:
            st.error(f"Failed to clean metadata: {result.stdout}")

# Function to create tab content
def create_tab_content(tab_name):
    with st.container():
        # Clean metadata when changing tabs
        if st.session_state.get('current_tab') != tab_name:
            # Determine tab type and clean corresponding test folder
            tab_type = "stud" if tab_name.lower() == "stud detection" else \
                      "dimension" if tab_name.lower() == "dimension classification" else "brick"
            clean_test_images_metadata(tab_type)
            st.session_state['current_tab'] = tab_name

        col2, col1 = st.columns([1, 1])
        
        with col1:
            
            st.subheader("📤 Input Your Own Image")
            
            # Drag and drop zone
            uploaded_file = st.file_uploader(
                f"📎 Drag and drop an image for {tab_name}", 
                type=['jpg', 'jpeg', 'png'], 
                key=f"uploader_{tab_name}"
            )

            # Reset gallery selection when upload state changes
            file_key = f'uploaded_{tab_name}'
            if uploaded_file is not None and file_key not in st.session_state['uploaded_images']:
                # New upload - clear gallery selection
                for key in list(st.session_state.keys()):
                    if key.startswith(f'gallery_{tab_name}'):
                        del st.session_state[key]
            elif uploaded_file is None and file_key in st.session_state['uploaded_images']:
                # Upload removed - clear upload state, selection, and force gallery refresh
                del st.session_state['uploaded_images'][file_key]
                # Clear all gallery-related state to force complete re-render
                for key in list(st.session_state.keys()):
                    if key.startswith(f'gallery_{tab_name}'):
                        del st.session_state[key]
                # Force streamlit to re-render this component
                st.rerun()
            
            # Handle file upload
            if uploaded_file is not None:
                # Convert uploaded file to bytes once
                if file_key not in st.session_state['uploaded_images']:
                    bytes_data = uploaded_file.getvalue()
                    img = Image.open(BytesIO(bytes_data))
                    # Resize if too large while maintaining aspect ratio
                    if img.width > 800 or img.height > 800:
                        ratio = min(800/img.width, 800/img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    # Save as JPEG with good quality
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG', quality=95)
                    st.session_state['uploaded_images'][file_key] = img_byte_arr.getvalue()
            
            # Determine which type of images to load
            if tab_name.lower() == "stud detection":
                tab_type = "stud"
            elif tab_name.lower() == "dimension classification":
                tab_type = "dimension"
            else:
                tab_type = "brick"

            st.subheader("🖼️ Select from Gallery")
            test_images, captions = load_test_images(tab_type)
            
            selected_image = None
            if not test_images and not uploaded_file:
                st.warning(f"No test images found for {tab_name}")
            else:
                # Prepare gallery images
                gallery_images = []
                gallery_captions = []
                
                # Add uploaded image if present
                if uploaded_file is not None and file_key in st.session_state['uploaded_images']:
                    uploaded_bytes = st.session_state['uploaded_images'][file_key]
                    gallery_images.append(Image.open(BytesIO(uploaded_bytes)))
                    gallery_captions.append("Uploaded Image")
                
                # Add test images
                for img_path in test_images:
                    try:
                        pil_img = Image.open(img_path)
                        if pil_img.width > 800 or pil_img.height > 800:
                            ratio = min(800/pil_img.width, 800/pil_img.height)
                            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
                            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                        gallery_images.append(pil_img)
                    except Exception as e:
                        st.warning(f"Could not load image {img_path}: {str(e)}")
                        continue
                gallery_captions.extend(captions)

                # Show gallery with all images
                if gallery_images:
                    # Create a unique key that changes when upload state changes
                    gallery_key = f"gallery_{tab_name}"
                    if uploaded_file is not None:
                        gallery_key += "_with_upload"
                    
                    selected_idx = image_select(
                        "Select an image from the gallery",
                        images=gallery_images,
                        captions=gallery_captions,
                        use_container_width=True,
                        return_value="index",
                        key=gallery_key
                    )
                    
                    # Handle selection
                    if selected_idx is not None:
                        if uploaded_file is not None and selected_idx == 0:
                            # Selected uploaded image
                            selected_image = BytesIO(st.session_state['uploaded_images'][file_key])
                            caption = "Uploaded Image"
                        else:
                            # Selected test image
                            img_idx = selected_idx - 1 if uploaded_file is not None else selected_idx
                            selected_image = test_images[img_idx]
                            caption = captions[img_idx]

            
            

        with col2:
            st.subheader("✨ Results")
            result_placeholder = st.empty()

            if st.button("🔄 Process Image", key=f"process_{tab_name}", use_container_width=True):
                if selected_image:
                    with st.spinner('Processing image...'):
                        # Get image data from either upload or gallery
                        if isinstance(selected_image, (BytesIO, st.runtime.uploaded_file_manager.UploadedFile)):
                            file_key = f'uploaded_{tab_name}'
                            if file_key in st.session_state['uploaded_images']:
                                result_image, metadata = process_single_image(
                                    BytesIO(st.session_state['uploaded_images'][file_key]), 
                                    tab_name
                                )
                            else:
                                st.error("Upload data not found in session state")
                                return
                        else:
                            result_image, metadata = process_single_image(selected_image, tab_name)

                        if result_image:
                            # For Dimension Classification tab, just show the composite image
                            if tab_name == "Dimension Classification":
                                result_placeholder.image(
                                    result_image, 
                                    caption="Full Analysis Result", 
                                    use_container_width=True
                                )

                                # Store in virtual outputs with metadata
                                img_byte_arr = BytesIO()
                                result_image.save(img_byte_arr, format='JPEG')
                                st.session_state['virtual_outputs'][tab_name]['images'].append({
                                    'name': f"result_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
                                    'data': img_byte_arr.getvalue(),
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                    'metadata': metadata
                                })
                            else:
                                result_placeholder.image(result_image, caption="Processed Result", use_container_width=True)

                                # Original metadata display logic
                                if metadata:
                                    # 1. Top level metrics remain as before
                                    metric_cols = st.columns(3)
                                    
                                    # Detection count
                                    if "boxes_coordinates" in metadata:
                                        detection_count = len(metadata["boxes_coordinates"])
                                        metric_cols[0].metric("Detections", detection_count)
                                    
                                    # Dimension if available
                                    if "dimension" in metadata:
                                        metric_cols[1].metric("Dimension", metadata["dimension"])
                                    
                                    # Processing time
                                    if "speed" in metadata:
                                        speed = metadata["speed"]
                                        total_time = sum(speed.values())
                                        metric_cols[2].metric("Processing Time", f"{total_time:.2f}s")
                                    elif "processing_time" in metadata:
                                        time_str = metadata["processing_time"]
                                        if isinstance(time_str, str) and time_str.endswith('s'):
                                            time_str = time_str[:-1]  # Remove 's' suffix
                                        metric_cols[2].metric("Processing Time", f"{float(time_str):.2f}s")

                                    # 2. Detection Details Table
                                    if "boxes_coordinates" in metadata:
                                        st.subheader("Detection Details")
                                        detection_data = []
                                        for idx, box_info in metadata["boxes_coordinates"].items():
                                            if isinstance(box_info, dict):
                                                coords = box_info.get("coordinates", [])
                                                conf = box_info.get("confidence", None)
                                                class_id = box_info.get("class", None)
                                                detection_data.append({
                                                    "ID": idx,
                                                    "Coordinates": f"[{', '.join(f'{c:.1f}' for c in coords)}]",
                                                    "Confidence": f"{conf:.2%}" if conf else "N/A",
                                                    "Class": class_id if class_id is not None else "N/A"
                                                })
                                        if detection_data:
                                            st.dataframe(detection_data, use_container_width=True)

                                    # 3. Speed Metrics Table
                                    if "speed" in metadata:
                                        st.subheader("Processing Speed")
                                        speed_df = [
                                            {"Stage": "Preprocessing", "Time (s)": f"{metadata['speed'].get('preprocess', 0):.3f}"},
                                            {"Stage": "Inference", "Time (s)": f"{metadata['speed'].get('inference', 0):.3f}"},
                                            {"Stage": "Postprocessing", "Time (s)": f"{metadata['speed'].get('postprocess', 0):.3f}"},
                                            {"Stage": "Total", "Time (s)": f"{sum(metadata['speed'].values()):.3f}"}
                                        ]
                                        st.dataframe(speed_df, use_container_width=True)

                                    # 4. System Information
                                    sys_info = {}
                                    for key in ["os_full_version_name", "processor", "architecture", "hostname"]:
                                        if key in metadata:
                                            sys_info[key.replace("_", " ").title()] = metadata[key]
                                    
                                    if sys_info:
                                        with st.expander("System Information"):
                                            for key, value in sys_info.items():
                                                st.text(f"{key}: {value}")

                                    # 5. Additional Information
                                    extra_info = {}
                                    for key in ["mode", "TimesScanned", "Repository", "message"]:
                                        if key in metadata:
                                            extra_info[key] = metadata[key]
                                    
                                    if extra_info:
                                        with st.expander("Additional Information"):
                                            for key, value in extra_info.items():
                                                # Format key for display
                                                display_key = " ".join(word.capitalize() for word in key.split("_"))
                                                st.text(f"{display_key}: {value}")
                else:
                    st.warning("Please upload an image or select a test image first.")

            # Show virtual folder contents
            if st.session_state['virtual_outputs'][tab_name]['images']:
                with st.expander("📂 Previous Results", expanded=False):
                    st.subheader("📅 Processing History")
                    for idx, img_data in enumerate(reversed(st.session_state['virtual_outputs'][tab_name]['images'])):
                        # Create a container for each result
                        with st.container():
                            # Image and metadata in columns
                            img_col, meta_col = st.columns([2, 1])
                            with img_col:
                                st.image(img_data['data'], caption=img_data['name'], use_container_width=True)
                            
                            with meta_col:
                                st.caption(f"Processed: {img_data['timestamp']}")
                                
                                # Display key metrics
                                metadata = img_data.get('metadata', {})
                                if metadata:
                                    if "boxes_coordinates" in metadata:
                                        st.metric("Detections", len(metadata["boxes_coordinates"]))
                                    if "dimension" in metadata:
                                        st.metric("Dimension", metadata["dimension"])
                                    
                                # Remove button aligned to the right
                                if st.button("Remove", key=f"remove_{tab_name}_{idx}", use_container_width=True):
                                    st.session_state['virtual_outputs'][tab_name]['images'].remove(img_data)
                                    st.rerun()
                            
                            # Add a separator between results
                            if idx < len(st.session_state['virtual_outputs'][tab_name]['images']) - 1:
                                st.markdown("---")

            # Help text
            with st.expander("❓ How to use this demo"):
                st.markdown(f"""
                1. 📤 Upload your own image or select one from the gallery
                2. 🔄 Click 'Process Image' to run {tab_name.lower()}
                3. 📊 View the results and detection metadata
                """)
            
            # Show selected image preview at the bottom of col2
            if selected_image:
                st.markdown("---")
                st.subheader("🖼️ Selected Image")
                if isinstance(selected_image, st.runtime.uploaded_file_manager.UploadedFile):
                    st.image(selected_image, caption="Uploaded Image", use_container_width=True)
                elif isinstance(selected_image, BytesIO):
                    img_bytes = st.session_state['uploaded_images'][f'uploaded_{tab_name}']
                    st.image(img_bytes, caption="Uploaded Image", use_container_width=True)
                else:
                    st.image(selected_image, caption=caption if 'caption' in locals() else "Selected Image", use_container_width=True)

# Main app
def main():
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Brick Detection", 
        "🎯 Stud Detection", 
        "📏 Dimension Classification",
        "🎨 Multiclass DEMO"
    ])
    
    with tab1:
        create_tab_content("Brick Detection")
    
    with tab2:
        create_tab_content("Stud Detection")
    
    with tab3:
        create_tab_content("Dimension Classification")

    with tab4:
        create_tab_content("Multiclass DEMO")  # Add new tab
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit • [🔗 Source Code](https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision)")

if __name__ == "__main__":
    main()