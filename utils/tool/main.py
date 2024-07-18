import json
from collections import defaultdict
from pycocotools.coco import COCO

def count_category_frequencies(coco_annotation_file):
    # Load COCO annotation file
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    coco = COCO(coco_annotation_file)

    # Initialize category frequency dictionary
    category_freq = defaultdict(int)

    # Get all categories
    categories = coco.loadCats(coco.getCatIds())
    category_names = {cat['id']: cat['name'] for cat in categories}

    # Count frequencies
    for ann in coco_data['annotations']:
        category_id = ann['category_id']
        category_freq[category_id] += 1

    # Create result list with all categories
    result = []
    for cat_id in sorted(category_names):
        result.append((category_names[cat_id], category_freq[cat_id]))

    return result


def write_frequencies_to_file(category_frequencies, output_file):
    # Find the maximum length of category names for formatting
    max_length = max(len(category) for category, freq in category_frequencies)

    with open(output_file, 'w') as f:
        for category, freq in category_frequencies:
            f.write(f"{category.ljust(max_length)} : {freq}\n")


# Example usage
coco_annotation_file = './annotations/train.json'
output_file = './train.txt'

category_frequencies = count_category_frequencies(coco_annotation_file)
write_frequencies_to_file(category_frequencies, output_file)

print(f"Frequencies written to {output_file}")

