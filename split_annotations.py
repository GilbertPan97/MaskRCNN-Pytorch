import os
import argparse
import json
import random
from pycocotools.coco import COCO


def split_annotations(annotations_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Load the COCO formatted JSON file
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Get images and annotations data
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Get image IDs with annotations
    annotated_image_ids = set(ann['image_id'] for ann in annotations)

    # Filter out images with no annotations
    images_with_annotations = [img for img in images if img['id'] in annotated_image_ids]

    # Calculate the number of images
    num_images = len(images_with_annotations)

    # Calculate the number of images for each split
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = num_images - num_train - num_val

    # Shuffle the images
    random.shuffle(images_with_annotations)

    # Split the dataset
    train_images = images_with_annotations[:num_train]
    val_images = images_with_annotations[num_train:num_train + num_val]
    test_images = images_with_annotations[num_train + num_val:]

    # Get annotations data for each split
    train_annotations = [ann for ann in annotations if ann['image_id'] in [img['id'] for img in train_images]]
    val_annotations = [ann for ann in annotations if ann['image_id'] in [img['id'] for img in val_images]]
    test_annotations = [ann for ann in annotations if ann['image_id'] in [img['id'] for img in test_images]]

    # Construct new COCO dataset format
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }

    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories']
    }

    test_data = {
        'images': test_images,
        'annotations': test_annotations,
        'categories': coco_data['categories']
    }

    # Save as new JSON files
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)

    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f)


def main():
    parser = argparse.ArgumentParser(description='Split COCO annotations into train, validation, and test sets.')
    parser.add_argument('annotations_dir', type=str, help='Directory containing annotations.json file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of images for the training set (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of images for the validation set (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of images for the test set (default: 0.1)')
    args = parser.parse_args()

    annotations_file = os.path.join(args.annotations_dir, 'annotations.json')

    # Create output directory
    output_dir = args.annotations_dir
    os.makedirs(output_dir, exist_ok=True)

    # Split dataset and save as new JSON files
    split_annotations(annotations_file, output_dir, args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()


# Usuage commond:
# python split_annotations.py coco_dataset/annotations --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
