import cv2
import numpy as np
import time
import os

IMG_TXT_PATH = "../../data/list/Tu_indoor/train.txt"
ROOT_DIR_PATH = "../../data/Tu_indoor"
img_count = 0
class_colors = np.array([
                        [0, 0, 0],
                        [192, 0, 0],
                        [128, 64, 128],
                        [0, 0, 128],
                        [0, 64, 64],
                        [128, 128, 192],
                        [128, 0, 64],
                        [128, 128, 128],
                        ])
class_count = np.zeros(len(class_colors))

with open(IMG_TXT_PATH) as f:
    for img_paths in f.read().splitlines():
        img_path_list = img_paths.split(' ')
        train_L_path = img_path_list[1]
        print(train_L_path)
        img = cv2.imread(os.path.join(ROOT_DIR_PATH, train_L_path), cv2.IMREAD_COLOR)
        for i, color in enumerate(class_colors):
            target_mask = cv2.inRange(img, color, color)
            class_count[i] += cv2.countNonZero(target_mask)
        img_count += 1

np.set_printoptions(precision = 6, suppress = True)
class_mean = class_count / img_count
class_weights = class_mean / np.sum(class_mean)
print("class_weights is")
print(class_weights)

