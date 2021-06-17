#!/usr/bin/python3
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

import tools._init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import myinfer, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class DDRNet():
    def __init__(self):
        self.pub = rospy.Publisher('segmentated_image', Image, queue_size=1)

        segmentation_hz = rospy.get_param("ddr_net/segmentation_hz", 1)
        camera_fps = rospy.get_param("ddr_net/camera_fps", 30)
        self.cnt = 0
        self.border = camera_fps // segmentation_hz
        config_path = rospy.get_param("ddr_net/config_path", "/")
        model_pretrained_path= rospy.get_param("ddr_net/model_pretrained_path", "/")
        model_trained_path= rospy.get_param("ddr_net/model_trained_path", "/")
        self.image_width = rospy.get_param("ddr_net/image_width", 2048)
        self.image_height= rospy.get_param("ddr_net/image_height", 1024)

        parser = argparse.ArgumentParser(description='Train segmentation network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            default=config_path,
                            type=str)
        parser.add_argument('opts',
                            help="Modify config options using the command-line",
                            default=None,
                            nargs=argparse.REMAINDER)
        args = parser.parse_args()
        args.opts = []
        update_config(config, args)
        config.defrost()
        config.MODEL.PRETRAINED = model_pretrained_path
        config.freeze()

        logger, final_output_dir, _ = create_logger(
            config, args.cfg, 'test')

#        logger.info(pprint.pformat(args))
#        logger.info(pprint.pformat(config))

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        # build model
        if torch.__version__.startswith('1'):
            module = eval('models.'+config.MODEL.NAME)
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        self.model = eval('models.'+config.MODEL.NAME +
                     '.get_seg_model')(config)

        model_state_file = model_trained_path
        logger.info('=> loading model from {}'.format(model_state_file))
        pretrained_dict = torch.load(model_state_file)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        gpus = list(config.GPUS)
        self.model = nn.DataParallel(self.model, device_ids=gpus).cuda()
        self.model.eval()

        # prepare data
        test_size = (self.image_height, self.image_width)
        self.test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                            root=config.DATASET.ROOT,
                            num_samples=None,
                            num_classes=config.DATASET.NUM_CLASSES,
                            multi_scale=False,
                            flip=False,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            base_size=config.TEST.BASE_SIZE,
                            crop_size=test_size,
                            downsample_rate=1)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.bridge = CvBridge()
        self.image_msg = Image()
        self.image_msg.height = self.image_height
        self.image_msg.width = self.image_width
        self.image_msg.encoding = "rgb8"
        self.image_msg.is_bigendian = False
        self.image_msg.step = 3 * self.image_width
        rospy.loginfo("complete loading DDRNet")


    def callback(self, msg):
        self.cnt += 1
        self.cnt %= self.border
        if self.cnt == 0:
            try:
                image_orig = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                print(e)

            start = time.time()
            image = cv2.resize(image_orig, dsize=(self.image_width, self.image_height))
            image = self.test_dataset.input_transform(image)
            image = self.transform(image)

            image = image.view(1, 3, self.image_height, self.image_width)

            pred = myinfer(config, 
                            self.test_dataset, 
                            image, 
                            self.model)

            self.image_msg.header.stamp = rospy.Time.now()
            self.image_msg.data = np.array(pred).tobytes()
            self.pub.publish(self.image_msg)

            try:
                imageMsg = self.bridge.cv2_to_imgmsg(pred, "bgr8")
                self.pub.publish(imageMsg)
            except Exception as e:
                print(e)

            end = time.time()
            rospy.loginfo('Mins: %lf' % (end-start))


if __name__ == '__main__':
    rospy.init_node('DDRNet')
    ddr = DDRNet()
    rospy.Subscriber("image_raw", Image, ddr.callback, queue_size=1)
    rospy.spin()
