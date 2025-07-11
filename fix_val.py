import os
import shutil

base_dir = '/home/gopalks/scripts/anu/ANN_SNN_QCFS/tiny-imagenet-200'
val_dir = os.path.join(base_dir, 'val')
val_images_dir = os.path.join(val_dir, 'images')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
val_output_dir = os.path.join(base_dir, 'val_split')

# Read annotations
annots = {}
with open(val_annotations_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split('\t')
        img_name, class_name = parts[0], parts[1]
        annots[img_name] = class_name

# Create class folders
os.makedirs(val_output_dir, exist_ok=True)
for class_name in set(annots.values()):
    os.makedirs(os.path.join(val_output_dir, class_name), exist_ok=True)

# Move images
for img_name, class_name in annots.items():
    src = os.path.join(val_images_dir, img_name)
    dst = os.path.join(val_output_dir, class_name, img_name)
    if os.path.exists(src):
        shutil.move(src, dst)

print("Validation images moved successfully.")
