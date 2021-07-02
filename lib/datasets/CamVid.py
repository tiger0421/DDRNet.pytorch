# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class CamVid(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=32,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=0, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(CamVid, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {-1: ignore_label, 0: ignore_label} 
        for s in range(1, 33):
            self.label_mapping[s] = s
        self.class_weights = torch.FloatTensor([
                                        0.00, 0.42, 0.30, 0.10,
                                        20.79, 4.15, 0.02, 0.01,
                                        1.04, 0.87, 1.55, 0.07,
                                        0.54, 0.00, 0.28, 0.37,
                                        0.56, 25.98, 0.28, 6.69,
                                        0.17, 18.04, 0.53, 0.01,
                                        0.33, 0.00, 10.76, 0.65,
                                        0.02, 0.94, 2.81, 1.72,
                                        ]).cuda()
        self.color_map = {
            "Animal"              : (64, 128, 64  ), 
            "Archway"             : (192, 0, 128  ), 
            "Bicyclist"           : (0, 128, 192  ), 
            "Bridge"              : (0, 128, 64   ), 
            "Building"            : (128, 0, 0    ), 
            "Car"                 : (64, 0, 128   ), 
            "CartLuggagePram"     : (64, 0, 192   ), 
            "Child"               : (192, 128, 64 ), 
            "Column_Pole"         : (192, 192, 128), 
            "Fence"               : (64, 64, 128  ), 
            "LaneMkgsDriv"        : (128, 0, 192  ), 
            "LaneMkgsNonDriv"     : (192, 0, 64   ), 
            "Misc_Text"           : (128, 128, 64 ), 
            "MotorcycleScooter"   : (192, 0, 192  ), 
            "OtherMoving"         : (128, 64, 64  ), 
            "ParkingBlock"        : (64, 192, 128 ), 
            "Pedestrian"          : (64, 64, 0    ), 
            "Road"                : (128, 64, 128 ), 
            "RoadShoulder"        : (128, 128, 192), 
            "Sidewalk"            : (0, 0, 192    ), 
            "SignSymbol"          : (192, 128, 128), 
            "Sky"                 : (128, 128, 128), 
            "SUVPickupTruck"      : (64, 128, 192 ), 
            "TrafficCone"         : (0, 0, 64     ), 
            "TrafficLight"        : (0, 64, 64    ), 
            "Train"               : (192, 64, 128 ), 
            "Tree"                : (128, 128, 0  ), 
            "Truck_Bus"           : (192, 128, 192), 
            "Tunnel"              : (64, 0, 64    ), 
            "VegetationMisc"      : (192, 192, 0  ), 
            "Void"                : (0, 0, 0      ), 
            "Wall"                : (64, 192, 0   ), 
        }


    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'CamVid',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label_rgb = cv2.imread(os.path.join(self.root,'CamVid',item["label"]),
                           cv2.IMREAD_COLOR)
        label_rgb = cv2.resize(label_rgb, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        label = np.zeros((label_rgb.shape[0], label_rgb.shape[1]))
        # convert rgb to label number
        for i, v in enumerate(self.color_map.values()):
            label[np.all(label_rgb == np.array(v), axis=2)] = i

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def mymulti_scale_inference(self, config, model, image, scales=[1], flip=False):
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        for scale in scales:
            new_img = self.mymulti_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False,
                                           align_corners=config.MODEL.ALIGN_CORNERS)
            height, width = image.shape[2:]

            if scale <= 1.0:
                pred = self.inference(config, model, image, flip)
                pred = pred[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                pred = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        pred[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                pred = pred / count
                pred = pred[:,:,:height,:width]

            pred = F.interpolate(
                pred, (height, width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
        return pred



    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
    def convert_pred_to_color(self, pred):
        palette = self.get_palette(256)
        pred = np.asarray(torch.argmax(pred, dim=1).cpu(), dtype=np.uint8)
        pred = self.convert_label(pred[0], inverse=True)
        pred_image = Image.fromarray(pred)
        pred_image.putpalette(palette)
        return pred_image.convert("RGB")

