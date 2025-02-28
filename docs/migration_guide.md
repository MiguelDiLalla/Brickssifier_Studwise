# Migration Guide: From Monolithic to Modular Structure

This guide explains how to migrate from the original monolithic `model_utils.py` structure to the new modular architecture.

## Architecture Overview

The new modular structure separates the original `model_utils.py` into specialized modules:

```
utils/
├── __init__.py            # Package initialization and exports
├── config_utils.py        # Configuration and resource loading
├── detection_utils.py     # Brick and stud detection
├── exif_utils.py          # EXIF metadata handling
├── classification_utils.py # Dimension classification
├── visualization_utils.py  # Image annotation and visualization
├── pipeline_utils.py      # End-to-end workflows
├── metadata_utils.py      # Result metadata extraction
├── rich_utils.py          # Rich text formatting utilities
└── model_utils_legacy.py  # Legacy wrapper for backward compatibility
```

## Function Mapping

| Old Path | New Path |
|----------|----------|
| `model_utils.setup_utils` | `config_utils.setup_utils` |
| `model_utils.read_exif` | `exif_utils.read_exif` |
| `model_utils.write_exif` | `exif_utils.write_exif` |
| `model_utils.clean_exif_metadata` | `exif_utils.clean_exif_metadata` |
| `model_utils.extract_metadata_from_yolo_result` | `metadata_utils.extract_metadata_from_yolo_result` |
| `model_utils.detect_bricks` | `detection_utils.detect_bricks` |
| `model_utils.detect_studs` | `detection_utils.detect_studs` |
| `model_utils.classify_dimensions` | `classification_utils.classify_dimensions` |
| `model_utils.annotate_scanned_image` | `visualization_utils.annotate_scanned_image` |
| `model_utils.read_detection` | `visualization_utils.read_detection` |
| `model_utils.run_full_algorithm` | `pipeline_utils.run_full_algorithm` |

## Migration Examples

### Before:

```python
from utils.model_utils import detect_bricks, detect_studs, run_full_algorithm

# Run detection
result = detect_bricks("image.jpg")
```

### After:

```python
# Option 1: Import from specific modules
from utils.detection_utils import detect_bricks, detect_studs
from utils.pipeline_utils import run_full_algorithm

# Option 2: Import from utils package (recommended)
from utils import detect_bricks, detect_studs, run_full_algorithm

# Run detection (same API)
result = detect_bricks("image.jpg")
```

## Backward Compatibility

For backward compatibility, you can continue using the old imports through the `model_utils_legacy.py` wrapper:

```python
from utils.model_utils_legacy import detect_bricks, detect_studs
```

However, this will display a deprecation warning encouraging migration to the new structure.

## Config Management

The global configuration dictionary is now accessible from `config_utils.py`:

```python
from utils.config_utils import config

# Access configuration
repo_url = config["REPO_URL"]
```

## Benefits of the New Structure

1. **Improved Organization**: Each module has a clear, focused purpose
2. **Better Code Navigation**: Smaller files are easier to read and understand
3. **Enhanced Maintainability**: Changes to one aspect don't affect others
4. **Proper Dependency Management**: Clear separation of concerns
5. **Better Documentation**: Each module has its own specialized documentation
6. **Easier Testing**: Modules can be tested independently
7. **Future Expansion**: New functionality can be added to the appropriate module

## Additional Notes

- All original functionality is preserved
- APIs remain the same for direct function calls
- Use the utils/__init__.py imports for the cleanest code
