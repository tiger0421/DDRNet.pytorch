import cv2
import numpy as np
import os

orig_path = '/home/ubuntu/share/cam_lidar/Tu_indoor/aisle02_dir/aisle02_img_000000.png'
target_path = "/home/ubuntu/share/cam_lidar/Tu_indoor/aisle01_dir"
target_files = os.listdir(target_path)
target_imgs = [f for f in target_files if os.path.isfile(os.path.join(target_path, f))]

img_orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
lower = np.array([128, 0, 64])
upper = np.array([128, 0, 64])
img_mask = cv2.inRange(img_orig, lower, upper)
img_mask_inv = cv2.bitwise_not(img_mask)
img_mask_color = cv2.bitwise_and(img_orig, img_orig, mask=img_mask)

for img_name in target_imgs:
    target_img = cv2.imread(os.path.join(target_path, img_name), cv2.IMREAD_COLOR)
    target_mask_color = cv2.bitwise_and(target_img, target_img, mask=img_mask_inv)
    result = cv2.add(target_mask_color, img_mask_color)
    print(os.path.join(target_path,  img_name[:-3]) + "png")
    cv2.imwrite(os.path.join(target_path, img_name[:-3] + "png"), result)
