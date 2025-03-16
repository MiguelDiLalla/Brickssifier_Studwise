import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from pathlib import Path

def augment_dataset(dataset_path: str, output_path: str):
    """
    Applies data augmentation to the dataset and saves the augmented images and labels.
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        ToTensorV2()
    ])

    image_dir = Path(dataset_path) / 'images'
    label_dir = Path(dataset_path) / 'labels'
    output_image_dir = Path(output_path) / 'images'
    output_label_dir = Path(output_path) / 'labels'

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for image_file in image_dir.glob('*.jpg'):
        image = cv2.imread(str(image_file))
        label_file = label_dir / (image_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        with open(label_file, 'r') as f:
            labels = f.readlines()

        augmented = transform(image=image)
        augmented_image = augmented['image']

        output_image_path = output_image_dir / image_file.name
        output_label_path = output_label_dir / label_file.name

        cv2.imwrite(str(output_image_path), augmented_image)
        with open(output_label_path, 'w') as f:
            f.writelines(labels)

    print(f"Augmented dataset saved to {output_path}")
