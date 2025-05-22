import os
import random
import shutil

# Set your base path
base_path = r"D:\AI Project\data_small"

# Original images and labels directories
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

# New train and val directories
train_images_dir = os.path.join(images_dir, "train")
val_images_dir = os.path.join(images_dir, "val")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")

# Create directories if they don't exist
for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
    os.makedirs(dir_path, exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

# Shuffle images
random.shuffle(image_files)

# Split 80/20
train_count = int(0.8 * len(image_files))
train_files = image_files[:train_count]
val_files = image_files[train_count:]

# Function to move files
def move_files(file_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
    for file_name in file_list:
        # Move image
        shutil.move(os.path.join(src_img_dir, file_name), os.path.join(dest_img_dir, file_name))
        
        # Move corresponding label
        label_name = os.path.splitext(file_name)[0] + ".txt"
        if os.path.exists(os.path.join(src_lbl_dir, label_name)):
            shutil.move(os.path.join(src_lbl_dir, label_name), os.path.join(dest_lbl_dir, label_name))
        else:
            print(f"Label not found for: {file_name}")

# Move train and val files
move_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
move_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)

print(f"✅ Done! Moved {len(train_files)} to train and {len(val_files)} to val.")
