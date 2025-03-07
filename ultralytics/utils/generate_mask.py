#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

import torch
from loguru import logger

import cv2

import numpy as np


def generate_mask(batch):       ##根据标注框产生掩膜 ，标注框区域变成白色，背景黑色
    num_img = batch['img'].shape[0]
    mask_batch = torch.zeros((num_img, 1, batch['img'].shape[2], batch['img'].shape[3]))

    for i in range (num_img):
        img = batch['img'][i, :, :, :]
        batch_idx = batch['batch_idx'].tolist()
        gt_idx = [m for m,n in enumerate(batch_idx) if n == i]
        labels = batch['bboxes'][gt_idx]
        height, width = img.shape[1:]
        mask = np.zeros((1, height, width))
        num_box = labels.shape[0]

        labels = labels.numpy()

        for j in range (num_box):
            up_x = (labels[j, 0] - labels[j, 2] / 2) * width  # 左上角x
            up_y = (labels[j, 1] - labels[j, 3] / 2) * height  # 左上角y
            button_x = (labels[j, 0] + labels[j, 2] / 2) * width  # 右下角x
            button_y= (labels[j, 1] + labels[j, 3] / 2) * height  # 右下角y

            up_x = int(up_x+0.5)
            up_y = int(up_y+0.5)
            button_x = int(button_x+0.5)
            button_y = int(button_y+0.5)

            mask[0, up_y:button_y, up_x:button_x] = 255


        mask_batch[i, :, :, :] = torch.tensor(mask)

    return mask_batch

def generate_mask_expand(batch):    ##根据标注框产生掩膜 ，一幅图像中标注框从上到下进行扩展，大致标出公路区域
    num_img = batch['img'].shape[0]
    mask_batch = torch.zeros((num_img, 1, batch['img'].shape[2], batch['img'].shape[3]))
    mask_expand_batch = torch.zeros((num_img, 1, batch['img'].shape[2], batch['img'].shape[3]))

    for i in range (num_img):
        img = batch['img'][i, :, :, :]
        batch_idx = batch['batch_idx'].tolist()
        gt_idx = [m for m,n in enumerate(batch_idx) if n == i]
        labels = batch['bboxes'][gt_idx]
        height, width = img.shape[1:]
        mask = np.zeros((1, height, width))
        mask_expand = np.zeros((1, height, width))
        num_box = labels.shape[0]

        labels = labels.numpy()

        for j in range (num_box):
            up_x = (labels[j, 0] - labels[j, 2] / 2) * width  # 左上角x
            up_y = (labels[j, 1] - labels[j, 3] / 2) * height  # 左上角y
            button_x = (labels[j, 0] + labels[j, 2] / 2) * width  # 右下角x
            button_y= (labels[j, 1] + labels[j, 3] / 2) * height  # 右下角y

            up_x = int(up_x+0.5)
            up_y = int(up_y+0.5)
            button_x = int(button_x+0.5)
            button_y = int(button_y+0.5)

            mask[0, up_y:button_y, up_x:button_x] = 255

        min_x = width
        max_x = 0
        row_value =  np.zeros((width))
        start = 0
        max_row_num = 0

        for r in range(height-1,0,-1):
            row_value = mask[0, r, :]
            if np.sum(row_value) != 0:
                max_row_num = r
                break

        for r in range (max_row_num):
            row_value = mask[0, r, :]
            if np.sum(row_value) != 0 :
                start = 1
                idx_255 = np.where(row_value == 255)
                x_left = np.min(idx_255)
                x_right = np.max(idx_255)

                if x_left < min_x:
                    min_x = x_left
                if x_right > max_x:
                    max_x = x_right

            if start==1:
                mask_expand[0, r, min_x : max_x] = 255

        mask_batch[i, :, :, :] = torch.tensor(mask)
        mask_expand_batch[i, :, :, :] = torch.tensor(mask_expand)

    return mask_batch, mask_expand_batch

def generate_mask_damage(batch):       ##将每个标注框内的图像进行二值处理产生掩膜，裂缝区域变成白色，背景黑色
    num_img = batch['img'].shape[0]
    mask_batch = torch.zeros((num_img, 1, batch['img'].shape[2], batch['img'].shape[3]))

    for i in range (num_img):
        img = batch['img'][i, :, :, :]
        batch_idx = batch['batch_idx'].tolist()
        gt_idx = [m for m,n in enumerate(batch_idx) if n == i]
        labels = batch['bboxes'][gt_idx]
        height, width = img.shape[1:]
        mask = np.zeros((1, height, width))
        num_box = labels.shape[0]

        labels = labels.numpy()
        img = img.numpy()

        for j in range (num_box):
            up_x = (labels[j, 0] - labels[j, 2] / 2) * width  # 左上角x
            up_y = (labels[j, 1] - labels[j, 3] / 2) * height  # 左上角y
            button_x = (labels[j, 0] + labels[j, 2] / 2) * width  # 右下角x
            button_y= (labels[j, 1] + labels[j, 3] / 2) * height  # 右下角y

            up_x = int(up_x+0.5)
            up_y = int(up_y+0.5)
            button_x = int(button_x+0.5)
            button_y = int(button_y+0.5)

            img_patch = img[:, up_y:button_y, up_x:button_x]
            img_patch = np.transpose(img_patch, [1, 2, 0])
            gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)  # 要二值化图像，要先进行灰度化处理

            average_gray_value = np.mean(gray)
            max_gray_value = np.max(gray)
            gray[gray>average_gray_value] = average_gray_value

            # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)

            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)  # 膨胀
            binary = liantong_remove_noise(binary, 50)  # 连通域去噪

            mask[0, up_y:button_y, up_x:button_x] = binary

        mask_batch[i, :, :, :] = torch.tensor(mask)

    return mask_batch


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_image_list(path):       #返回需要进行测试的文件
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):      #遍历文件夹，file_name_list里存储所有文件名
        for filename in file_name_list:         #依次读取文件名
            apath = os.path.join(maindir, filename)     #将目录和文件名连接，组成完整的路径
            ext = os.path.splitext(apath)[1]        #返回文件的扩展名
            if ext in IMAGE_EXT:            #看扩展名是否是图像类型
                image_names.append(apath)
    return image_names

from skimage.measure import label

def make_parser():
    parser = argparse.ArgumentParser("Visual label")                 #创建参数解析器

    parser.add_argument(        #调用的图片或者视频的路径
        "--path", default="E:/dataset/CNRDD/train/2022/ttt", help="path to images or video"
        # "--path", default="E:/dataset/RDD2020_union/train/union/ttt", help="path to images or video"
    )

    return parser



if __name__ == "__main__":
    args = make_parser().parse_args()       #解析参数

    # from_img_to_mask_damage(args)
    # from_img_to_mask_road(args)