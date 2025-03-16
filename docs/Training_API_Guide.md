# LEGO Bricks ML Vision - Training Models API

The Training Models API provides a comprehensive suite for training machine learning models for LEGO brick and stud detection. This module handles data preparation, augmentation, and model training through an intuitive interface.

## Core Components

### TrainingPipeline

Main orchestrator for the complete training workflow.

```python
from training import TrainingPipeline

# Initialize pipeline for brick detection
pipeline = TrainingPipeline(mode='bricks')

# Run training with custom parameters
results = pipeline.run(
    dataset_path='path/to/dataset',
    epochs=50,
    batch_size=16,
    device='cuda',
    augmentation_count=2
)
```

Supported modes:
- `bricks`: Training for brick detection
- `studs`: Training for stud detection
- `multiclass`: Combined training for both

### Single-Class Training

Specialized training for either brick or stud detection:

```python
from training import SingleClassTrainer

trainer = SingleClassTrainer(mode='bricks')
results = trainer.train(
    dataset_path='path/to/dataset',
    model_path='yolov8n.pt',
    epochs=50,
    batch_size=16
)
```

### Multi-Class Training

Combined training for detecting both bricks and studs:

```python
from training import MultiClassTrainer

trainer = MultiClassTrainer()
results = trainer.train(
    dataset_path='path/to/dataset',
    model_path='yolov8n.pt',
    epochs=50,
    batch_size=16
)
```

### Data Augmentation

Customizable image augmentation pipeline:

```python
from training import DataAugmenter

augmenter = DataAugmenter(mode='bricks')
stats = augmenter.augment_dataset(
    images_dir='path/to/images',
    labels_dir='path/to/labels',
    augmentation_count=2
)
```

Default augmentation techniques:
- Random rotation (90°)
- Horizontal/Vertical flips
- Brightness/Contrast adjustments
- Noise injection (Gaussian, ISO, Multiplicative)

## Data Preparation Functions

```python
from training import validate_dataset, create_dataset_structure, split_dataset

# Validate dataset structure
validate_dataset(mode='bricks')

# Create training directory structure
dataset_dir = create_dataset_structure(mode='bricks', base_path='path/to/dataset')

# Split dataset into train/val/test
split_info = split_dataset(mode='bricks')
```

## Error Handling and Validation

### Error Handling Decorator
The training module provides a consistent error handling decorator:

```python
from training.utils import handle_training_errors

@handle_training_errors
def your_training_function():
    """Will have consistent error handling and logging"""
    pass
```

### Project Structure Validation
The training module validates the project structure before execution:

```python
from training.validation import get_training_root, validate_dataset_path

# Get and validate project root
root = get_training_root()  # Raises TrainingRootError if invalid

# Validate dataset path
dataset_path = validate_dataset_path('path/to/dataset')
```

Required project structure:
```
LEGO_Bricks_ML_Vision/
├── training/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── training_pipeline.py
│   ├── single_class_trainer.py
│   └── multiclass_trainer.py
├── presentation/
│   └── Datasets_Compress/
└── cache/
```

### Error Types

- `TrainingError`: Base exception for training-related errors
- `TrainingRootError`: Invalid project structure or paths
- `FileNotFoundError`: Missing dataset files or configurations

Error handling example:
```python
from training.validation import TrainingRootError

try:
    pipeline = TrainingPipeline(mode='bricks')
    results = pipeline.run(dataset_path='path/to/dataset')
except TrainingRootError as e:
    print(f"Invalid project structure: {str(e)}")
except Exception as e:
    print(f"Training failed: {str(e)}")
```

## Technical Details

### Model Architecture
- Based on YOLOv8
- Customized for LEGO brick detection
- Supports single and multi-class detection

### Performance Optimization
- Automatic device selection (CPU/GPU)
- Batch size optimization
- Memory-efficient data loading

### Requirements
- PyTorch
- Ultralytics YOLO
- Albumentations
- OpenCV
- NumPy

## Version Information
- Current Version: 0.1.0
- Author: Miguel DiLalla

## Module Reference

### Data Preparation Module (`data_preparation.py`)

#### `unzip_dataset(zip_path: Path, extract_dir: Path, force_extract: bool = False) -> Path`
Extracts the dataset from a zip file for training pipeline use.
- **zip_path**: Path to the zipped dataset
- **extract_dir**: Target extraction directory
- **force_extract**: Whether to overwrite existing extraction
- **Returns**: Path to extracted dataset

#### `validate_dataset(mode: str) -> tuple`
Validates the structure and contents of a training dataset.
- **mode**: Training mode ('bricks', 'studs', 'multiclass')
- **Returns**: Tuple of (images_dir, labels_dir, valid_pairs)

#### `create_dataset_structure(mode: str, repo_root: Path) -> Path`
Creates YOLO-compatible directory structure for training.
- **mode**: Dataset mode
- **repo_root**: Repository root path
- **Returns**: Path to created dataset directory

#### `split_dataset(mode: str, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1) -> dict`
Splits dataset into training, validation and test sets.
- **mode**: Dataset mode
- **train_ratio**: Proportion for training set
- **val_ratio**: Proportion for validation set
- **test_ratio**: Proportion for test set
- **Returns**: Dictionary with split file lists

### Augmentation Module (`augmentation.py`)

#### DataAugmenter Class
```python
class DataAugmenter:
    def __init__(self, mode: str):
        """Initialize augmenter for specific detection mode"""
    
    def augment_dataset(self, images_dir: Path, labels_dir: Path, 
                       augmentation_count: int = 2) -> Dict[str, int]:
        """Perform dataset augmentation with statistics tracking"""
        
    def _create_transform_pipeline(self) -> A.Compose:
        """Create Albumentations transformation pipeline"""
```

Augmentation pipeline includes:
- Geometric: Random rotation, flips
- Intensity: Brightness, contrast, gamma
- Noise: Gaussian, ISO, multiplicative
- YOLO bbox coordinate handling

#### MultiClassTrainer (`multiclass_trainer.py`)
```python
class MultiClassTrainer:
    def __init__(self):
        """Initialize trainer for combined detection"""
    
    def extract_class_names(self, labels_folder: Path) -> list:
        """Extract class names from batch inference metadata JSON
        
        Args:
            labels_folder: Path containing batch_inference_metadata.json
            
        Returns:
            Ordered list of class names matching YOLO indices
            
        Example metadata JSON:
        {
          "config": {
            "classes": {
              "0": "1x1",
              "1": "2x1",
              "2": "3x1",
              ...
            }
          }
        }
        """
```

### Dataset Metadata Format
The training pipeline expects a metadata JSON file in the labels directory:

```json
{
  "config": {
    "classes": {
      "0": "1x1",
      "1": "2x1",
      "2": "3x1",
      // ... additional class mappings
    }
  }
}
```

This metadata file:
- Must be named `batch_inference_metadata.json`
- Located in the labels directory
- Contains class index to name mappings
- Used to configure YOLO training

#### SingleClassTrainer (`single_class_trainer.py`)
```python
class SingleClassTrainer:
    def __init__(self, mode: str):
        """Initialize trainer for brick/stud detection"""
    
    def train(self, dataset_path: Path, model_path: str, epochs: int = 50,
              batch_size: int = 16, device: str = 'auto', **kwargs):
        """Execute single-class optimized training"""
```

#### TrainingPipeline (`training_pipeline.py`)
```python
class TrainingPipeline:
    def __init__(self, mode: str):
        """Initialize complete training workflow"""
        
    def run(self, dataset_path: Path, epochs: int = 50, batch_size: int = 16,
            device: str = 'auto', augmentation_count: int = 2):
        """Execute end-to-end training pipeline"""
        
    def _cleanup_intermediate_files(self, dataset_dir: Path):
        """Clean temporary training files"""
```

### Error Handling (`exceptions.py`)

Custom exceptions for graceful error handling:
- `TrainingError`: Base training exception
- `DatasetError`: Dataset validation/processing errors
- `AugmentationError`: Image augmentation errors

### Utility Functions (`utils.py`)

#### `handle_training_errors`
Decorator for consistent error handling across training pipeline:
```python
@handle_training_errors
def your_training_function():
    """Will have consistent error handling"""
```

## Configuration

### Dataset Structure
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### YAML Configuration
```yaml
path: .
train: train/images
val: val/images
test: test/images
nc: 1  # number of classes
names: ['brick']  # or ['stud'] or ['1x1', '2x1', '3x1', '2x2', '4x1', '3x2', '6x1', '4x2', '8x1', '10x1', '6x2', '12x1', '4x4', '8x2']
```

## Error Codes and Messages

Common error codes and their meanings:
- `E001`: Dataset validation failed
- `E002`: Augmentation pipeline error
- `E003`: Training initialization failed
- `E004`: GPU memory allocation error
- `E005`: Model checkpoint error

## Best Practices

1. Dataset Preparation:
   - Use consistent image sizes
   - Ensure proper labeling format (YOLO)
   - Maintain balanced class distribution

2. Training Configuration:
   - Start with default hyperparameters
   - Adjust batch size based on available memory
   - Use validation set to monitor training

3. Augmentation:
   - Consider domain-specific augmentations
   - Monitor augmentation quality
   - Maintain aspect ratios for LEGO bricks

## Testing Module

### Test Structure
The training module includes comprehensive test coverage using pytest:

```python
tests/training_module/
├── conftest.py           # Shared fixtures
├── test_augmentation.py  # Data augmentation tests
├── test_data_preparation.py # Dataset processing tests
├── test_multiclass_trainer.py # Multi-class training tests
├── test_single_class_trainer.py # Single-class training tests
├── test_training_pipeline.py # End-to-end pipeline tests
├── test_training_utils.py # Utility function tests
└── test_validation.py    # Path validation tests
```

### Common Test Fixtures
Available in `conftest.py`:
```python
@pytest.fixture
def mock_project_root(tmp_path):
    """Creates mock project structure"""
    
@pytest.fixture
def mock_dataset(tmp_path):
    """Provides sample dataset structure"""
    
@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    
@pytest.fixture
def sample_image_array():
    """Sample image data"""
    
@pytest.fixture
def mock_yolo_results():
    """Mock YOLO detection results"""
```

### Running Tests
Execute the test suite:
```bash
# Run all training module tests
pytest tests/training_module/

# Run specific test file
pytest tests/training_module/test_training_pipeline.py

# Run with coverage report
pytest --cov=training tests/training_module/
```

### Key Test Cases

#### Pipeline Tests
- Complete workflow execution
- Error handling and cleanup
- Configuration validation
- Device selection

#### Trainer Tests
- Model initialization
- Training parameter validation
- Class name extraction
- Dataset compatibility

#### Data Preparation Tests
- Dataset structure validation
- Train/val/test splitting
- File pair matching
- Empty file detection

#### Augmentation Tests
- Transform pipeline creation
- Image/label augmentation
- Statistics tracking
- Error cases

### Mocking Strategy
The tests use pytest's monkeypatch and unittest.mock for:
- File system operations
- YOLO model interactions
- Logger calls
- External dependencies

Example:
```python
def test_train_with_mock_yolo(monkeypatch):
    mock_model = Mock()
    monkeypatch.setattr('ultralytics.YOLO', mock_model)
    
    trainer = SingleClassTrainer(mode="brick")
    trainer.train(dataset_path="path/to/dataset")
    
    mock_model.assert_called_once()
```

### Test Configuration
Example pytest.ini:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks integration tests
addopts = --strict-markers -ra
```
