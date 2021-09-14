import os
import glob
import random

root_dir = "../../data/Tu_indoor"
train_path = "train"
train_L_path = "trainannot_rgb"
list_dir = "../../data/list/Tu_indoor"
seed = 0
random.seed(seed)

# make list about images for train
train_files = os.listdir(os.path.join(root_dir, train_path))
train_dir = [d for d in train_files if os.path.isdir(os.path.join(root_dir, train_path, d))]
train_images = []
for d in train_dir:
    train_files = os.listdir(os.path.join(root_dir, train_path, d))
    tmp = [os.path.join(d, os.path.splitext(f)[0]) for f in train_files if os.path.isfile(os.path.join(root_dir, train_path, d, f))]
    tmp = [s for s in tmp if "DS_Store" not in s]
    train_images += tmp
_, train_ext = os.path.splitext(train_files[0])

# make list about annotated images
train_L_files = os.listdir(os.path.join(root_dir, train_path))
train_L_dir = [d for d in train_L_files if os.path.isdir(os.path.join(root_dir, train_L_path, d))]
train_L_images = []
for d in train_L_dir:
    train_L_files = os.listdir(os.path.join(root_dir, train_L_path, d))
    tmp = [os.path.join(d, os.path.splitext(f)[0]) for f in train_L_files if os.path.isfile(os.path.join(root_dir, train_L_path, d, f))]
    tmp = [s for s in tmp if "DS_Store" not in s]
    train_L_images += tmp
_, train_L_ext = os.path.splitext(train_L_files[0])

# extract common images and shuffle
train_names = list(set(train_images) & set(train_L_images))
train_names.sort()
random.shuffle(train_names)

# write train.txt and val.txt
tmp = sorted(train_names[:100])
train_list_path = os.path.join(list_dir, "train.txt")
with open(train_list_path, "w") as output:
    for img_name in tmp:
        content = os.path.join(train_path, img_name) + train_ext + " " + os.path.join(train_L_path, img_name) + train_L_ext + "\n"
        output.write(content)

tmp = sorted(train_names[100:])
train_L_list_path = os.path.join(list_dir, "val.txt")
with open(train_L_list_path, "w") as output:
    for img_name in tmp:
        content = os.path.join(train_path, img_name) + train_ext + " " + os.path.join(train_L_path, img_name) + train_L_ext + "\n"
        output.write(content)

