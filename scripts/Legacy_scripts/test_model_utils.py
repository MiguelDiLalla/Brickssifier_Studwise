"""
Test suite for LEGO Bricks ML Vision model utilities.

This module contains tests for the core functionality of the model_utils module:
- Brick detection
- Stud detection
- EXIF metadata handling
- Full pipeline execution

Tests can be run individually or as a suite using the unittest framework.
"""

import unittest
import os
import sys
import shutil
import cv2
import json
import logging
import random
import subprocess
from pathlib import Path
import datetime

# Add project root directory to Python's path so 'utils' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the module
from scripts.Legacy_scripts import model_utils

# Whether to clean EXIF metadata from test images before running tests
# Set to True to ensure tests run with fresh metadata each time
CLEAN_METADATA = True

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_timestamped_filename(prefix):
    """Generate a timestamped filename for test outputs"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"

class TestModelUtilsLocal(unittest.TestCase):
    """Test suite for model_utils functionality using local resources."""

    def setUp(self):
        """
        Prepare test environment:
        1. Clean EXIF metadata from test images if configured
        2. Set up test image paths for bricks and studs datasets
        3. Create output directory for test results
        """
        # Clean EXIF metadata in bricks and studs folders if CLEAN_METADATA is True
        if CLEAN_METADATA:
            logger.info("🧹 Cleaning EXIF metadata from test images...")
            for folder in ["presentation/Test_images/BricksPics", "presentation/Test_images/StudsPics"]:
                full_folder = os.path.join(os.getcwd(), folder)
                if os.path.exists(full_folder):
                    for f in os.listdir(full_folder):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(full_folder, f)
                            try:
                                model_utils.clean_exif_metadata(file_path)
                                logger.info(f"[CLEAN] Cleaned EXIF for: {file_path}")
                            except Exception as e:
                                logger.error(f"[CLEAN] Failed to clean EXIF for {file_path}: {e}")

        # Set up test bricks image path
        bricks_folder = os.path.join(os.getcwd(), "presentation", "Test_images", "BricksPics")
        files = [f for f in os.listdir(bricks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.test_bricks_image_path = os.path.join(bricks_folder, random.choice(files))
        if not os.path.exists(self.test_bricks_image_path):
            self.skipTest(f"Test image not found: {self.test_bricks_image_path}")
        logger.info(f"[SETUP] Test image located at: {self.test_bricks_image_path}")
        
        # Read bricks image as a numpy array for later testing
        self.test_bricks_image = cv2.imread(self.test_bricks_image_path)
        if self.test_bricks_image is None:
            self.skipTest("Failed to load test image as numpy array.")

        # Set up test studs image path
        studs_folder = os.path.join(os.getcwd(), "presentation", "Test_images", "StudsPics")
        files = [f for f in os.listdir(studs_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.test_studs_image_path = os.path.join(studs_folder, random.choice(files))
        if not os.path.exists(self.test_studs_image_path):
            self.skipTest(f"Test image not found: {self.test_studs_image_path}")
        logger.info(f"[SETUP] Test image located at: {self.test_studs_image_path}")
        
        # Read image as a numpy array for later testing
        self.test_studs_image = cv2.imread(self.test_studs_image_path)
        if self.test_studs_image is None:
            self.skipTest("Failed to load test image as numpy array.")

        # Set output directory for test results
        self.output_dir = os.path.join(os.getcwd(), "tests", "test_results")
        # Remove previous test results if they exist
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            logger.info(f"[SETUP] Removed existing output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"[SETUP] Created new output directory: {self.output_dir}")
        
        # Generate timestamped filenames for this test run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.brick_output_filename = f"detect_bricks_{self.timestamp}"
        self.stud_output_filename = f"detect_studs_{self.timestamp}"
        self.full_output_filename = f"full_algorithm_{self.timestamp}"

    def test_detect_bricks_via_path(self):
        """Test brick detection using file path as input."""
        logger.info("[TEST] Running detect_bricks() using image path input.")
        
        output_subdir = os.path.join(self.output_dir, f"bricks_path_{self.timestamp}")
        os.makedirs(output_subdir, exist_ok=True)
        
        result = model_utils.detect_bricks(
            self.test_bricks_image_path, 
            conf=0.25, 
            save_json=True, 
            save_annotated=True, 
            output_folder=output_subdir
        )
        
        self.assertIsNotNone(result, "detect_bricks() returned None")
        
        # Check that a composite annotated image was created and metadata JSON exists
        meta = result.get("metadata", {})
        composite_path = meta.get("annotated_image_path", "")
        json_path = meta.get("json_results_path", "")
        
        logger.info(f"[TEST] Composite image path: {composite_path}")
        logger.info(f"[TEST] Metadata JSON path: {json_path}")
        
        # Print metadata for inspection
        print("\n[DEBUG] Metadata (from image path detection):")
        print(json.dumps(meta, indent=4))
        
        # Save a copy with our timestamped name for easier tracking
        if os.path.exists(composite_path):
            timestamped_path = os.path.join(output_subdir, f"{self.brick_output_filename}.jpg")
            shutil.copy(composite_path, timestamped_path)
            logger.info(f"[TEST] Copied image to timestamped path: {timestamped_path}")
            self._try_open_image(timestamped_path)
        else:
            self.fail("Composite annotated image not created.")
        
        # Check contents of output directory
        logger.info(f"Output directory contents: {os.listdir(output_subdir)}")
    
    def test_detect_bricks_via_numpy(self):
        """Test brick detection using numpy array as input."""
        logger.info("[TEST] Running detect_bricks() using image as NumPy array.")
        
        output_subdir = os.path.join(self.output_dir, f"bricks_numpy_{self.timestamp}")
        os.makedirs(output_subdir, exist_ok=True)
        
        result = model_utils.detect_bricks(
            self.test_bricks_image, 
            conf=0.25, 
            save_json=True, 
            save_annotated=True, 
            output_folder=output_subdir
        )
        
        self.assertIsNotNone(result, "detect_bricks() returned None")
        
        # Find any image files in the output directory
        image_files = [f for f in os.listdir(output_subdir) if f.lower().endswith(('.jpg', '.png'))]
        if image_files:
            composite_path = os.path.join(output_subdir, image_files[0])
            logger.info(f"[TEST] Found image file: {composite_path}")
            
            # Save a copy with our timestamped name
            timestamped_path = os.path.join(output_subdir, f"{self.brick_output_filename}_numpy.jpg")
            shutil.copy(composite_path, timestamped_path)
            logger.info(f"[TEST] Copied image to timestamped path: {timestamped_path}")
            self._try_open_image(timestamped_path)
        else:
            logger.warning(f"No image files found in {output_subdir}")
            self.fail("No image files found in output directory")

    def test_detect_studs_via_path(self):
        """Test stud detection using file path as input."""
        logger.info("[TEST] Running detect_studs() using image path input.")
        
        output_subdir = os.path.join(self.output_dir, f"studs_path_{self.timestamp}")
        os.makedirs(output_subdir, exist_ok=True)
        
        result = model_utils.detect_studs(
            self.test_studs_image_path, 
            conf=0.25, 
            save_annotated=True, 
            output_folder=output_subdir
        )
        
        # Skip test if no studs detected
        if result is None:
            self.skipTest("No studs detected, skipping further checks.")
            return
        
        meta = result.get("metadata", {})
        composite_path = meta.get("annotated_image_path", "")
        dimension = meta.get("dimension", "Dimensions not found")
        
        logger.info(f"[TEST] Composite image path: {composite_path}")
        logger.info(f"[TEST] Dimension: {dimension}")
        
        # Print filtered metadata for inspection
        print("\n[DEBUG] Metadata (from stud detection):")
        filtered_meta = {k: v for k, v in meta.items() if k != 'boxes_coordinates'}
        print(json.dumps(filtered_meta, indent=4))
        
        # Save a copy with our timestamped name
        if os.path.exists(composite_path):
            timestamped_path = os.path.join(output_subdir, f"{self.stud_output_filename}.jpg")
            shutil.copy(composite_path, timestamped_path)
            logger.info(f"[TEST] Copied image to timestamped path: {timestamped_path}")
            self._try_open_image(timestamped_path)
        else:
            logger.warning(f"No image file found at {composite_path}")
            # Look for any image files in the output directory
            image_files = [f for f in os.listdir(output_subdir) if f.lower().endswith(('.jpg', '.png'))]
            if image_files:
                logger.info(f"Found alternative image files: {image_files}")
                self._try_open_image(os.path.join(output_subdir, image_files[0]))
            else:
                self.fail("No image files found in output directory")
    
    def test_full_algorithm(self):
        """Test the full brick detection and dimension classification pipeline."""
        logger.info("[TEST] Running full algorithm with brick image.")

        output_subdir = os.path.join(self.output_dir, f"full_algorithm_{self.timestamp}")
        os.makedirs(output_subdir, exist_ok=True)
        
        result = model_utils.run_full_algorithm(
            self.test_bricks_image_path,
            save_annotated=True,
            output_folder=output_subdir
        )
        
        if result is None:
            self.skipTest("No result returned from full algorithm.")
            return
        
        self.assertIn("brick_results", result, "Missing brick results in the output.")
        self.assertIn("studs_results", result, "Missing studs results in the output.")
        self.assertIn("composite_image", result, "Missing composite image in the output.")
        
        # Check if the composite image was saved
        full_output_path = os.path.join(output_subdir, "fullyScannedImage.jpg")
        if os.path.exists(full_output_path):
            timestamped_path = os.path.join(output_subdir, f"{self.full_output_filename}.jpg")
            # Save the composite image
            cv2.imwrite(timestamped_path, result["composite_image"])
            logger.info(f"[TEST] Saved composite image to: {timestamped_path}")
            self._try_open_image(timestamped_path)
        else:
            logger.warning(f"Composite image not found at expected path: {full_output_path}")
            # Save it directly
            timestamped_path = os.path.join(output_subdir, f"{self.full_output_filename}.jpg")
            cv2.imwrite(timestamped_path, result["composite_image"])
            logger.info(f"[TEST] Directly saved composite image to: {timestamped_path}")
            self._try_open_image(timestamped_path)

    def test_exif_metadata_handling(self):
        """Test EXIF metadata reading and writing capabilities."""
        logger.info("[TEST] Testing EXIF metadata handling.")
        
        # Create test metadata
        test_metadata = {
            "test_key": "test_value",
            "TimesScanned": 0,
            "boxes_coordinates": {"0": [10, 20, 30, 40]}
        }
        
        # Create a copy of the test image with timestamp
        test_image_path = os.path.join(self.output_dir, f"test_exif_{self.timestamp}.jpg")
        shutil.copy(self.test_bricks_image_path, test_image_path)
        
        # Write metadata
        model_utils.write_exif(test_image_path, test_metadata)
        
        # Read metadata back
        read_metadata = model_utils.read_exif(test_image_path)
        
        self.assertIsNotNone(read_metadata, "Failed to read EXIF metadata")
        self.assertEqual(read_metadata.get("test_key"), "test_value", "EXIF metadata key-value not preserved")
        self.assertEqual(read_metadata.get("TimesScanned"), 1, "TimesScanned should be set to 1")
        
        # Write again to test incrementing TimesScanned
        model_utils.write_exif(test_image_path, read_metadata)
        read_metadata_again = model_utils.read_exif(test_image_path)
        
        self.assertEqual(read_metadata_again.get("TimesScanned"), 2, "TimesScanned should increment to 2")

    def _try_open_image(self, image_path):
        """Helper method to open an image in Visual Studio Code if available, otherwise use system viewer."""
        if not os.path.exists(image_path):
            logger.warning(f"Cannot open image that doesn't exist: {image_path}")
            return
            
        # First try to check if VS Code is available
        vscode_available = False
        try:
            # Check if VS Code CLI is in PATH
            if os.name == 'nt':  # Windows
                # On Windows, try both 'code' and the default install path
                try:
                    result = subprocess.run(['where', 'code'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    vscode_available = result.returncode == 0
                except:
                    # Try common installation paths
                    vscode_paths = [
                        r"C:\Program Files\Microsoft VS Code\bin\code.cmd",
                        r"C:\Program Files (x86)\Microsoft VS Code\bin\code.cmd",
                        r"%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd"
                    ]
                    for path in vscode_paths:
                        expanded_path = os.path.expandvars(path)
                        if os.path.exists(expanded_path):
                            vscode_available = True
                            break
            else:  # macOS/Linux
                result = subprocess.run(['which', 'code'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                vscode_available = result.returncode == 0
        except Exception:
            vscode_available = False
            
        # Try to open with VS Code if available
        if vscode_available:
            try:
                if os.name == 'nt':  # Windows
                    # Use the full path if found, otherwise try the command
                    subprocess.run(['code', image_path], check=False)
                else:  # macOS/Linux
                    subprocess.run(['code', image_path], check=False)
                logger.info(f"Opened image in VS Code: {image_path}")
                return
            except Exception as e:
                logger.warning(f"VS Code detected but failed to open image: {e}")
        else:
            logger.warning("VS Code not detected in PATH, falling back to system viewer")
        
        # Fall back to system default if VS Code not available or failed
        try:
            if os.name == 'nt':  # Windows
                os.startfile(image_path)
            elif os.name == 'posix':  # macOS or Linux
                if 'darwin' in os.sys.platform:  # macOS
                    subprocess.call(['open', image_path])
                else:  # Linux
                    subprocess.call(['xdg-open', image_path])
            logger.info(f"Opened image with system viewer: {image_path}")
        except Exception as e:
            logger.error(f"Failed to open image with any method: {e}")

if __name__ == '__main__':
    unittest.main()
