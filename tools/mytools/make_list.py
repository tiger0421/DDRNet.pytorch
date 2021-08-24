import os
import random

root_dir = "../data/Tu_indoor"
train_path = "train"
train_L_path = "trainannot_rgb"
list_dir = "../data/list/Tu_indoor"
seed = 0
random.seed(seed)

# make list about images for train
train_files = os.listdir(os.path.join(root_dir, train_path))
train_dir = [d for d in train_files if os.path.isdir(os.path.join(root_dir, train_path, d))]
train_images = []
for d in train_dir:
    train_files = os.listdir(os.path.join(root_dir, train_path, d))
    tmp = [f for f in train_files if os.path.isfile(os.path.join(root_dir, train_path, d, f))]
    try:
        train_images.remove(".DS_Store")
    except ValueError:
        pass
    train_images += tmp

# make list about annotated images
train_L_files = os.listdir(os.path.join(root_dir, train_path))
train_L_dir = [d for d in train_L_files if os.path.isdir(os.path.join(root_dir, train_L_path, d))]
train_L_images = []
for d in train_dir:
    train_L_files = os.listdir(os.path.join(root_dir, train_path, d))
    tmp = [f for f in train_L_files if os.path.isfile(os.path.join(root_dir, train_path, d, f))]
    try:
        train_images.remove(".DS_Store")
    except ValueError:
        pass
    train_L_images += tmp

# extract common images and shuffle
train_names = list(set(train_images) & set(train_L_images))
train_names.sort()
random.shuffle(train_names)

# write train.txt and val.txt
tmp = sorted(train_names[:100])
train_list_path = os.path.join(list_dir, "train.txt")
with open(train_list_path, "w") as output:
    for img_name in tmp:
        content = os.path.join(train_path, img_name) + " " + os.path.join(train_L_path, img_name) + "\n"
        output.write(content)

tmp = sorted(train_names[100:])
train_L_list_path = os.path.join(list_dir, "val.txt")
with open(train_L_list_path, "w") as output:
    for img_name in tmp:
        content = os.path.join(train_path, img_name) + " " + os.path.join(train_L_path, img_name) + "\n"
        output.write(content)

