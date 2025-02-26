import unittest
import os
import shutil
import cv2
import json
import logging
from utils import model_utils
import random
import subprocess

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestModelUtilsLocal(unittest.TestCase):
    def setUp(self):
        # # Set up test bricks image path
        # bricks_folder = os.path.join(os.getcwd(), "presentation", "Test_images", "BricksPics")
        # files = [f for f in os.listdir(bricks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # self.test_bricks_image_path = os.path.join(bricks_folder, random.choice(files))
        # if not os.path.exists(self.test_bricks_image_path):
        #     self.skipTest(f"Test image not found: {self.test_bricks_image_path}")
        # logging.info(f"[SETUP] Test image located at: {self.test_bricks_image_path}")
        
        # # Read bricks image as a numpy array for later testing
        # self.test_bricks_image = cv2.imread(self.test_bricks_image_path)
        # if self.test_bricks_image is None:
        #     self.skipTest("Failed to load test image as numpy array.")

        #------------------------------------------

        # Set up test studs image path
        studs_folder = os.path.join(os.getcwd(), "presentation", "Test_images", "StudsPics")
        files = [f for f in os.listdir(studs_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.test_studs_image_path = os.path.join(studs_folder, random.choice(files))
        if not os.path.exists(self.test_studs_image_path):
            self.skipTest(f"Test image not found: {self.test_studs_image_path}")
        logging.info(f"[SETUP] Test image located at: {self.test_studs_image_path}")
        
        # Read image as a numpy array for later testing
        self.test_studs_image = cv2.imread(self.test_studs_image_path)
        if self.test_studs_image is None:
            self.skipTest("Failed to load test image as numpy array.")
        
        self.test_studs_specific_image_path = os.path.join(os.getcwd(), "presentation", "Test_images", "StudsPics", "image_28_LegoBrick_0_c88.jpg")
        self.test_studs_specific_image = cv2.imread(self.test_studs_specific_image_path)
        # ----------------------------------------------

        # Set output directory for test results
        self.output_dir = os.path.join(os.getcwd(), "tests", "test_results")
        # Remove previous test results if they exist
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            logging.info(f"[SETUP] Removed existing output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"[SETUP] Created new output directory: {self.output_dir}")

    # def test_detect_bricks_via_path(self):
    #     logging.info("[TEST] Running detect_bricks() using image path input.")
    #     # Call detect_bricks by passing the image path
    #     result = model_utils.detect_bricks(
    #         self.test_image_path, 
    #         model=None,             # Should default to loaded brick model from local mode
    #         conf=0.25, 
    #         save_json=True, 
    #         save_annotated=True, 
    #         output_folder=self.output_dir
    #     )
    #     # Check that a composite annotated image was created and metadata JSON exists
    #     meta = result.get("metadata", {})
    #     composite_path = meta.get("annotated_image_path", "")
    #     json_path = meta.get("json_results_path", "")
    #     logging.info(f"[TEST] Composite image path: {composite_path}")
    #     logging.info(f"[TEST] Metadata JSON path: {json_path}")
    #     # NEW: Print full metadata dictionary for inspection
    #     print("\n[DEBUG] Metadata (from image path detection):")
    #     print(json.dumps(meta, indent=4))
    #     self.assertTrue(os.path.exists(composite_path), "Composite annotated image not created.")
    #     self.assertTrue(os.path.exists(json_path), "Metadata JSON file not created.")
    
    # def test_detect_bricks_via_numpy(self):
    #     logging.info("[TEST] Running detect_bricks() using image as NumPy array.")
    #     # Call detect_bricks by passing the numpy array (not a file path)
    #     result = model_utils.detect_bricks(
    #         self.test_image, 
    #         model=None, 
    #         conf=0.25, 
    #         save_json=True, 
    #         save_annotated=True, 
    #         output_folder=self.output_dir
    #     )
    #     meta = result.get("metadata", {})
    #     composite_path = meta.get("annotated_image_path", "")
    #     json_path = meta.get("json_results_path", "")
    #     logging.info(f"[TEST] Composite image path (numpy input): {composite_path}")
    #     logging.info(f"[TEST] Metadata JSON path (numpy input): {json_path}")
    #     # NEW: Print full metadata dictionary for inspection
    #     print("\n[DEBUG] Metadata (from NumPy array detection):")
    #     # print(json.dumps(meta, indent=4))
    #     self.assertTrue(os.path.exists(composite_path), "Composite annotated image not created (via NumPy).")
    #     self.assertTrue(os.path.exists(json_path), "Metadata JSON file not created (via NumPy).")
    
    # def test_render_metadata(self):
    #     logging.info("[TEST] Testing render_metadata() function directly.")
    #     # Render metadata panel on the test image using dummy metadata
    #     dummy_meta = {"dummy": True, "info": "Render test"}
    #     rendered_img = model_utils.render_metadata(self.test_image, dummy_meta)
    #     self.assertIsNotNone(rendered_img, "render_metadata() returned None.")
    #     # Save rendered metadata image in the output bricks_folder
    #     render_output_path = os.path.join(self.output_dir, "rendered_metadata.jpg")
    #     cv2.imwrite(render_output_path, rendered_img)
    #     logging.info(f"[TEST] Rendered metadata image saved at: {render_output_path}")
    #     self.assertTrue(os.path.exists(render_output_path), "Rendered metadata image file was not created successfully.")

    # def test_retrieve_exif_data(self):
    #     '''
    #     cheks how many images from the bricks_folder have exif UserComment data
    #     then pick randomly again and store anotated version using the info inside the boxes key

    #     '''
    #     bricks_folder = os.path.join(os.getcwd(), "presentation", "Test_images", "BricksPics")
    #     files = [f for f in os.listdir(bricks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #     images_with_exif = []

    #     for file in files:
    #         image_path = os.path.join(bricks_folder, file)
    #         metadata = model_utils.read_exif(image_path)
    #         if metadata:
    #             images_with_exif.append(image_path)


    #     if images_with_exif:
    #         selected_image = random.choice(images_with_exif)
    #         logging.info(f"[TEST] Selected image for EXIF retrieval: {selected_image}")

    #     # now we retrieve bounding boxes from the exif data
    #     # metadata = model_utils.read_exif(selected_image)

    #     # Annotate the image
    #     annotated_image = model_utils.annotate_scanned_image(selected_image)
    #     self.assertIsNotNone(annotated_image, "annotate_scanned_image() returned None.")
    #     # Save rendered metadata image in the output bricks_folder
    #     rendered_output_path = os.path.join(self.output_dir, "annotated_image.jpg")
    #     cv2.imwrite(rendered_output_path, annotated_image)
    #     logging.info(f"[TEST] Annotated image saved at: {rendered_output_path}")
    #     self.assertTrue(os.path.exists(rendered_output_path), "Annotated image file was not created successfully.")

    # def test_read_exif_data(self):
    #     '''
    #     pick a random image from the bricks_folder and use the read_exif function to retrieve the exif data
    #     then prints the dictionary
    #     '''
    #     bricks_folder = os.path.join(os.getcwd(), "presentation", "Test_images", "BricksPics")
    #     files = [f for f in os.listdir(bricks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #     images_with_exif = []

    #     for file in files:
    #         image_path = os.path.join(bricks_folder, file)
    #         metadata = model_utils.read_exif(image_path)
    #         if metadata:
    #             images_with_exif.append(image_path)

    #     if images_with_exif:
    #         selected_image = random.choice(images_with_exif)
    #         logging.info(f"[TEST] Selected image for EXIF retrieval: {selected_image}")

    #         metadata = model_utils.read_exif(selected_image)
    #         print("\n[DEBUG] EXIF Metadata:")
    #         print(json.dumps(metadata, indent=4))
    #     else:
    #         logging.warning("⚠️ No images with EXIF data found in the bricks_folder.")


    def test_detect_studs_via_path(self):
        logging.info("[TEST] Running detect_bricks() using image path input.")
        # Call detect_bricks by passing the image path
        result = model_utils.detect_studs(
            self.test_studs_image_path, 
            model=None,             # Should default to loaded brick model from local mode
            conf=0.25, 
            save_annotated=True, 
            output_folder=self.output_dir
        )
        # Check that a composite annotated image was created and metadata JSON exists
        meta = result.get("metadata", {})
        composite_path = meta.get("annotated_image_path", "")

        logging.info(f"[TEST] Composite image path: {composite_path}")
        logging.info(f"[TEST] Dimension: {meta.get('dimension', '')}")
        # NEW: Print full metadata dictionary for inspection
        print("\n[DEBUG] Metadata (from image path detection):")
        filtered_meta = {k: v for k, v in meta.items() if k != 'boxes_coordinates'}
        print(json.dumps(filtered_meta, indent=4))
        # self.assertTrue(os.path.exists(composite_path), "Composite annotated image not created.")

        # if in visual studio, open the composite_path
        # Try to open the image in the default viewer if it exists
        if os.path.exists(composite_path):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(composite_path)
                elif os.name == 'posix':  # macOS or Linux
                    if 'darwin' in os.sys.platform:  # macOS
                        subprocess.call(['open', composite_path])
                    else:  # Linux
                        subprocess.call(['xdg-open', composite_path])
                logging.info(f"Opened annotated image: {composite_path}")
            except Exception as e:
                logging.error(f"Failed to open image: {e}")

    # def test_detect_studs_target_brick(self):


if __name__ == '__main__':
    unittest.main()
