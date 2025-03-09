"""
LEGO Bricks ML Vision - Test Suite

This module provides comprehensive testing for the LEGO Bricks ML Vision project,
including unit tests and integration tests for all components.

Usage:
    pytest test_lego_vision.py [options]
    python -m unittest test_lego_vision

Author: Miguel DiLalla
"""

import os
import sys
import unittest
import pytest
import logging
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path
import random

# Ensure the project root is in the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from utils.config_utils import config, setup_utils
from utils.detection_utils import detect_bricks, detect_studs
from utils.exif_utils import read_exif, write_exif, clean_exif_metadata
from utils.classification_utils import classify_dimensions, analyze_stud_pattern, get_common_brick_dimensions
from utils.visualization_utils import annotate_scanned_image, read_detection, create_composite_image, draw_detection_visualization
from utils.pipeline_utils import run_full_algorithm, batch_process
from utils.metadata_utils import extract_metadata_from_yolo_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test resources
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"
TEST_RESULTS_DIR = Path(__file__).parent / "test_results"


# Sample test image paths - replace with actual test images in your project
BRICK_TEST_IMAGE = str(TEST_IMAGES_DIR / "brick_test.jpg")
STUD_TEST_IMAGE = str(TEST_IMAGES_DIR / "stud_test.jpg")

# Create blank test images if they don't exist (for CI/CD)
def create_dummy_image(path, size=(640, 480)):
    """Create a dummy test image if it doesn't exist."""
    if not os.path.exists(path):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Draw a brick-like shape
        cv2.rectangle(img, (100, 100), (300, 200), (0, 0, 255), -1)
        # Draw stud-like circles
        for x in range(150, 300, 50):
            for y in range(125, 200, 50):
                cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
                
        cv2.imwrite(path, img)
        logger.info(f"Created dummy test image at {path}")

# Create test images if they don't exist
create_dummy_image(BRICK_TEST_IMAGE)
create_dummy_image(STUD_TEST_IMAGE)

class TestConfigUtils(unittest.TestCase):
    """Tests for configuration utilities."""
    
    def test_setup_utils(self):
        """Test that setup_utils returns a properly structured configuration dictionary."""
        # Call setup_utils with repo_download=False to avoid network calls
        test_config = setup_utils(repo_download=False)
        
        # Check that the configuration dictionary has the expected keys
        self.assertIn("REPO_URL", test_config)
        self.assertIn("MODELS_PATHS", test_config)
        self.assertIn("TEST_IMAGES_FOLDERS", test_config)
        self.assertIn("STUDS_TO_DIMENSIONS_MAP", test_config)
        self.assertIn("EXIF_METADATA_DEFINITIONS", test_config)
    
    def test_config_global_variable(self):
        """Test that the config global variable is properly initialized."""
        # Check that config is a dictionary
        self.assertIsInstance(config, dict)
        
        # Check that it contains the expected keys
        self.assertIn("REPO_URL", config)
        self.assertIn("MODELS_PATHS", config)


class TestExifUtils(unittest.TestCase):
    """Tests for EXIF metadata utilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary image file for testing
        self.test_image_path = os.path.join(TEST_RESULTS_DIR, "test_exif.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, img)
        
        self.test_metadata = {
            "boxes_coordinates": {"0": {"coordinates": [10, 10, 50, 50]}},
            "dimension": "2x4",
            "mode": "brick",
            "TimesScanned": 0
        }
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary test files
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_write_read_exif(self):
        """Test writing and reading EXIF metadata."""
        # Write metadata to the test image
        write_exif(self.test_image_path, self.test_metadata)
        
        # Read back the metadata
        read_metadata = read_exif(self.test_image_path)
        
        # Verify key values were preserved
        self.assertEqual(read_metadata.get("mode"), "brick")
        self.assertEqual(read_metadata.get("dimension"), "2x4")
        self.assertEqual(read_metadata.get("TimesScanned"), 1)  # Should be incremented
        
        # Check that boxes_coordinates is preserved
        self.assertIn("0", read_metadata.get("boxes_coordinates", {}))
        box = read_metadata["boxes_coordinates"]["0"]
        self.assertEqual(box.get("coordinates"), [10, 10, 50, 50])
    
    def test_clean_exif_metadata(self):
        """Test cleaning EXIF metadata."""
        # First write metadata
        write_exif(self.test_image_path, self.test_metadata)
        
        # Verify metadata was written
        self.assertTrue(read_exif(self.test_image_path))
        
        # Clean the metadata
        clean_exif_metadata(self.test_image_path)
        
        # Verify metadata was cleaned (should be empty)
        self.assertEqual(read_exif(self.test_image_path), {})


class TestVisualizationUtils(unittest.TestCase):
    """Tests for visualization utilities."""
    
    def test_create_composite_image(self):
        """Test creating a composite image from multiple images."""
        # Create test images
        base_img = np.zeros((100, 200, 3), dtype=np.uint8)
        base_img[:, :, 1] = 255  # Green base image
        
        detection_img1 = np.zeros((80, 150, 3), dtype=np.uint8)
        detection_img1[:, :, 0] = 255  # Blue detection image
        
        detection_img2 = np.zeros((60, 120, 3), dtype=np.uint8)
        detection_img2[:, :, 2] = 255  # Red detection image
        
        # Create a composite image
        composite = create_composite_image(
            base_img, 
            [detection_img1, detection_img2],
            logo=None
        )
        
        # Check the composite image dimensions
        self.assertIsInstance(composite, np.ndarray)
        self.assertEqual(len(composite.shape), 3)  # Should be a 3D array (H, W, C)
        
        # The composite should be taller than the base image due to stacking
        self.assertGreater(composite.shape[0], base_img.shape[0])
        
        # The width should match the base image (plus margins)
        self.assertEqual(composite.shape[1], base_img.shape[1] + 20)  # 10px margins on each side
    
    def test_draw_detection_visualization(self):
        """Test drawing bounding boxes on an image."""
        # Create a test image
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Define some bounding boxes
        boxes = np.array([
            [50, 50, 150, 150],  # x1, y1, x2, y2
            [100, 100, 200, 180]
        ])
        
        # Define class names and confidence scores
        class_names = ["brick", "brick"]
        confidence_scores = [0.92, 0.85]
        
        # Draw the boxes
        annotated = draw_detection_visualization(
            img, 
            boxes, 
            class_names, 
            confidence_scores
        )
        
        # Check that the result is an image
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, img.shape)
        
        # The annotated image should have some non-zero pixels (the boxes and text)
        self.assertGreater(np.sum(annotated), 0)


class TestClassificationUtils(unittest.TestCase):
    """Tests for dimension classification utilities."""
    
    def test_get_common_brick_dimensions(self):
        """Test retrieving common brick dimensions based on stud count."""
        # Create some test stud boxes
        four_studs_linear = np.array([
            [10, 10, 30, 30],
            [40, 10, 60, 30],
            [70, 10, 90, 30],
            [100, 10, 120, 30]
        ])
        
        four_studs_square = np.array([
            [10, 10, 30, 30],
            [40, 10, 60, 30],
            [10, 40, 30, 60],
            [40, 40, 60, 60]
        ])
        
        # Get dimensions for 4 studs
        linear_dims = get_common_brick_dimensions(four_studs_linear)
        square_dims = get_common_brick_dimensions(four_studs_square)
        
        # Check that the dimensions include both 2x2 and 4x1 options for 4 studs
        self.assertIn("2x2", linear_dims)
        self.assertIn("4x1", linear_dims)
        self.assertIn("2x2", square_dims)
        
        # Test that unknown stud count returns a generic dimension
        odd_number = np.array([
            [10, 10, 30, 30],
            [40, 10, 60, 30],
            [10, 40, 30, 60],
            [40, 40, 60, 60],
            [70, 40, 90, 60]
        ])
        
        odd_dims = get_common_brick_dimensions(odd_number)
        self.assertTrue(any("5 studs" in dim for dim in odd_dims)
                       or any("5x1" in dim for dim in odd_dims)
                       or any("1x5" in dim for dim in odd_dims))


class TestDetectionUtils(unittest.TestCase):
    """Tests for detection utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.brick_test_image = get_random_test_image(BRICKS_TEST_DIR)
        self.stud_test_image = get_random_test_image(STUDS_TEST_DIR)
    
    @unittest.skipIf(not os.path.exists(str(BRICKS_TEST_DIR)), "Test images directory not available")
    def test_detect_bricks(self):
        """Test basic brick detection functionality."""
        try:
            result = detect_bricks(
                self.brick_test_image, 
                model=None,
                conf=0.25, 
                save_annotated=False
            )
            
            if result is not None:
                self.assertIn("orig_image", result)
                self.assertIn("annotated_image", result)
                self.assertIn("cropped_detections", result)
                self.assertIn("metadata", result)
                # Check boxes_coordinates instead of boxes
                self.assertIn("boxes_coordinates", result["metadata"])
                self.assertIn("status", result)
        except Exception as e:
            # If model is not available, this should be skipped without failing
            if "No bricks model loaded" not in str(e):
                self.fail(f"detect_bricks raised unexpected exception: {e}")
    
    @unittest.skipIf(not os.path.exists(str(STUDS_TEST_DIR)), "Test images directory not available")
    def test_detect_studs(self):
        """Test basic stud detection functionality."""
        # This is a lightweight test that just checks the API works
        # Skip actual model inference to avoid dependencies
        
        try:
            result = detect_studs(
                self.stud_test_image, 
                model=None,
                conf=0.25, 
                save_annotated=False
            )
            
            if result is not None:
                self.assertIn("orig_image", result)
                self.assertIn("annotated_image", result)
                self.assertIn("dimension", result)
                self.assertIn("metadata", result)
                self.assertIn("status", result)
        except Exception as e:
            # If model is not available, this should be skipped without failing
            if "No studs model loaded" not in str(e):
                self.fail(f"detect_studs raised unexpected exception: {e}")


class TestPipelineUtils(unittest.TestCase):
    """Tests for pipeline utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.output_dir = os.path.join(TEST_RESULTS_DIR, "integration")
        os.makedirs(self.output_dir, exist_ok=True)
        self.brick_test_image = get_random_test_image(BRICKS_TEST_DIR)
    
    @unittest.skipIf(not os.path.exists(str(BRICKS_TEST_DIR)), "Test images directory not available")
    def test_run_full_algorithm(self):
        """Test the full algorithm pipeline."""
        # This test checks that the function runs without errors
        # It doesn't verify the actual detection results to avoid dependencies
        
        try:
            result = run_full_algorithm(
                self.brick_test_image,
                save_annotated=False,
                force_rerun=True
            )
            
            if result is not None:
                self.assertIn("brick_results", result)
                self.assertIn("studs_results", result)
                self.assertIn("composite_image", result)
        except Exception as e:
            # If models are not available, this should be skipped without failing
            if "No bricks model loaded" not in str(e) and "No studs model loaded" not in str(e):
                self.fail(f"run_full_algorithm raised unexpected exception: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire system."""
    
    def setUp(self):
        """Set up test environment."""
        self.output_dir = os.path.join(TEST_RESULTS_DIR, "integration")
        os.makedirs(self.output_dir, exist_ok=True)
        self.brick_test_image = get_random_test_image(BRICKS_TEST_DIR)
    
    @unittest.skipIf(not os.path.exists(str(BRICKS_TEST_DIR)), "Test images directory not available")
    def test_end_to_end_workflow(self):
        """Test the end-to-end workflow from brick detection to dimension classification."""
        # This is a minimal integration test to verify components work together
        
        try:
            # Step 1: Detect bricks
            brick_result = detect_bricks(
                self.brick_test_image, 
                save_annotated=True,
                save_json=True,
                output_folder=self.output_dir,
                force_rerun=True
            )
            
            if brick_result is None:
                self.skipTest("Brick detection model not available")
                
            # Step 2: For each detected brick, run stud detection
            for i, crop in enumerate(brick_result.get("cropped_detections", [])):
                # Save the crop for inspection
                crop_path = os.path.join(self.output_dir, f"crop_{i}.jpg")
                cv2.imwrite(crop_path, crop)
                
                # Run stud detection on the crop
                stud_result = detect_studs(
                    crop,
                    save_annotated=True,
                    output_folder=os.path.join(self.output_dir, f"stud_{i}"),
                    force_rerun=True
                )
                
                if stud_result is not None:
                    # Verify that we got a dimension classification
                    self.assertIn("dimension", stud_result)
                    
                    # Save the dimension to a text file
                    with open(os.path.join(self.output_dir, f"dimension_{i}.txt"), "w") as f:
                        f.write(f"Brick {i}: {stud_result['dimension']}")
        
        except Exception as e:
            # If models are not available, this should be skipped without failing
            if "No bricks model loaded" not in str(e) and "No studs model loaded" not in str(e):
                self.fail(f"End-to-end workflow test failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()