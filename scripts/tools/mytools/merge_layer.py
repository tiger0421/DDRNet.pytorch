import cv2
import numpy as np
import os

under_layer_path = '/home/ubuntu/share/cam_lidar/Tu_indoor/red2'
upper_layer_path = "/home/ubuntu/share/cam_lidar/Tu_indoor/aisle02_dir"
target_files = os.listdir(upper_layer_path)
target_imgs = [f for f in target_files if os.path.isfile(os.path.join(upper_layer_path, f))]
try:
    target_imgs.remove(".DS_Store")
except ValueError:
    pass

lower = np.array([0, 0, 128])
upper = np.array([0, 0, 128])
target_colors = np.array([
                        [0, 0, 0],
                        [192, 0, 0],
                        [128, 64, 128],
                        [0, 0, 128],
                        [0, 64, 64],
                        [128, 128, 192],
                        [128, 0, 64],
                        [128, 128, 128],
                        ])

for img_name in target_imgs:
    base_img = cv2.imread(os.path.join(under_layer_path, img_name), cv2.IMREAD_COLOR)
    result_img = np.zeros(base_img.shape, dtype=base_img.dtype)

    img_mask = cv2.inRange(base_img, lower, upper)
    img_mask_color = cv2.bitwise_and(base_img, base_img, mask=img_mask)
    result_img = cv2.add(result_img, img_mask_color)
    cv2.imwrite("result.png", result_img)

    target_img = cv2.imread(os.path.join(upper_layer_path, img_name), cv2.IMREAD_COLOR)
    for color in target_colors:
        img_mask = cv2.inRange(target_img, color, color)
        img_mask_inv = cv2.bitwise_not(img_mask)
        img_mask_color = cv2.bitwise_and(target_img, target_img, mask=img_mask)
        result_img = cv2.bitwise_and(result_img, result_img, mask=img_mask_inv)
        result_img = cv2.add(result_img, img_mask_color)

    print(os.path.join(upper_layer_path,  img_name[:-3]) + "png")
    cv2.imwrite(os.path.join(upper_layer_path, img_name[:-3] + "png"), result_img)
