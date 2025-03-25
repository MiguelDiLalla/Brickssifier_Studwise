# Monolithic vs Modular Implementation Comparison

This document provides a comparison between functions in the monolithic `LegoProjectUtils.py` script and their counterparts in the modular utils package.

## Function Mapping

| Monolithic Script (LegoProjectUtils.py) | Modular Implementation (utils/*) | Description |
|----------------------------------------|----------------------------------|-------------|
| `LegoVisionProject._load_default_config()` | `config_utils.init_config()` | Initialize default configuration settings |
| `LegoVisionProject.detect_bricks()` | `detection_utils.detect_bricks()` | Run brick detection on images |
| `LegoVisionProject.detect_studs()` | `detection_utils.detect_studs()` | Run stud detection on brick images |
| `LegoVisionProject._classify_brick_dimension()` | `classification_utils.classify_dimensions()` | Classify brick dimensions from stud pattern |
| `LegoVisionProject.process_labelme_to_yolo()` | `data_utils.convert_labelme_to_yolo()` | Convert LabelMe annotations to YOLO format |
| `LegoVisionProject.process_keypoints_to_boxes()` | `data_utils.convert_keypoints_to_boxes()` | Convert keypoint annotations to bounding boxes |
| `LegoVisionProject.read_exif()` | `exif_utils.read_metadata()` | Read EXIF metadata from images |
| `LegoVisionProject.clean_exif()` | `exif_utils.clean_metadata()` | Remove EXIF metadata from images |
| `LegoVisionProject.run_batch_inference()` | `batch_utils.process_batch_inference()` | Run batch processing on multiple images |
| `LegoVisionProject.visualize_detections()` | `visualization_utils.create_detection_grid()` | Create grid visualization of detections |
| `LegoVisionProject.visualize_dataset()` | `visualization_utils.create_dataset_preview()` | Visualize dataset with annotations |
| `LegoVisionProject.create_demo_dataset()` | `visualization_utils.create_conversion_demo()` | Generate annotation conversion demos |
| CLI Commands | `lego_cli.py` | Command-line interface implementations |

## Key Differences

1. **Organization**
   - Monolithic: All functionality in single class for simplified usage
   - Modular: Separated by functionality type for better maintenance

2. **Dependencies**
   - Monolithic: Direct imports of core libraries
   - Modular: Indirect dependencies through utility modules

3. **Configuration**
   - Monolithic: Configuration managed by class instance
   - Modular: Shared configuration through config_utils

4. **Progress Tracking**
   - Monolithic: Integrated Rich progress UI
   - Modular: Optional progress tracking via callback

5. **Error Handling**
   - Monolithic: Centralized error classes
   - Modular: Module-specific error handling

## Benefits of Each Approach

### Monolithic Script
- Easier to deploy - single file solution
- Direct access to all functionality
- Consistent UI/UX through class methods
- Simplified state management
- Reduced import complexity

### Modular Implementation
- Better code organization
- Easier to maintain individual components
- More flexible for future extensions
- Better for team development
- Enables selective feature usage

## Recommended Usage

- Use monolithic script for:
  - Quick deployments
  - Single-user scenarios
  - Standalone applications
  - Simple pipelines

- Use modular implementation for:
  - Complex projects
  - Team development
  - Custom integrations
  - Selective feature usage
  - Extended functionality needs