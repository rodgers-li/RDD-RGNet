#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import numpy as np

from matplotlib import pyplot as plt

CNRDD_CLASSES = ('S_1', 'S_3', 'S_4', 'S_5', 'S_6')

RDD2020_CLASSES = ('D00', 'D10', 'D20', 'D40')

KITTI_CLASSES = ('Pedestrian', 'Car', 'Truck', 'Van', 'bicycle','Person','Tram')
cityscapes_CLASS = ('person', 'rider', 'car', 'truck', 'bus','motorcycle','traffic sign','traffic light','bicycle')

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.384, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)



def make_parser():
    parser = argparse.ArgumentParser("Visual label")                 #创建参数解析器


    parser.add_argument(        #调用的图片或者视频的路径
        "--path", default="./img/VOC/JPEGImages", help="path to images or video"
    )

    return parser

def get_image_list(path):       #返回需要进行测试的文件
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):      #遍历文件夹，file_name_list里存储所有文件名
        for filename in file_name_list:         #依次读取文件名
            apath = os.path.join(maindir, filename)     #将目录和文件名连接，组成完整的路径
            ext = os.path.splitext(apath)[1]        #返回文件的扩展名
            if ext in IMAGE_EXT:            #看扩展名是否是图像类型
                image_names.append(apath)
    return image_names

def vis_gt(img, boxes, cls_ids, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}'.format(class_names[cls_id])
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.7 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]     #for RDD2020
        # txt_size = cv2.getTextSize(text, font, 1.0, 1)[0]           #for CNRDD

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)         #for RDD2020
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 7)            #for CNRDD

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - 1),
            (x0 + int(2*txt_size[0]) + 1, y0 - int(4*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.8, txt_color, thickness=2)     #for RDD2020
        # cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 2.0, txt_color, thickness=5)        #for CNRDD

    return img

def main(args):

    vis_folder = os.path.join("./", "vis_res")
    os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    current_time = time.localtime()

    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)

    if os.path.isdir(args.path):         #判断路径是否一个目录，isfile判断是否文件
        files = get_image_list(args.path)        #解析出目录下得所有文件

    files.sort()        #对文件进行排序


    for image_name in files:
        img = cv2.imread(image_name)  # 读取图片
        height, width = img.shape[:2]

        # img = torch.from_numpy(img).unsqueeze(0)    #将数组形式改为张量
        # img = img.float()

        label_name = image_name.replace('.jpg', ".txt")
        # label_name = image_name.replace('.png', ".txt")
        label_name = label_name.replace('val_image', "labels")

        # label_name = label_name.replace('t_foggy_beta_0.01', "t")
        # label_name = label_name.replace('t_foggy_beta_0.02', "t")
        # label_name = label_name.replace('t_foggy_beta_0.005', "t")

        labels = np.loadtxt(label_name)

        # print(labels.size)

        if labels.size == 0:
            vis_gt_res = img
        elif labels.size == 5:
            gt_boxes = labels[None, 1:].copy()
            gt_boxes[:, 0] = (labels[1] - labels[3]/2) * width      #左上角x
            gt_boxes[:, 1] = (labels[2] - labels[4]/2) * height     #左上角y
            gt_boxes[:, 2] = (labels[1] + labels[3]/2) * width      #右下角x
            gt_boxes[:, 3] = (labels[2] + labels[4]/2) * height     #右下角y
            cls = labels[None, 0]
            vis_gt_res = vis_gt(img, gt_boxes, cls, VOC_CLASSES)
        else:
            gt_boxes = labels[:, 1:].copy()
            gt_boxes[:, 0] = (labels[:, 1] - labels[:, 3]/2) * width      #左上角x
            gt_boxes[:, 1] = (labels[:, 2] - labels[:, 4]/2) * height     #左上角y
            gt_boxes[:, 2] = (labels[:, 1] + labels[:, 3]/2) * width      #右下角x
            gt_boxes[:, 3] = (labels[:, 2] + labels[:, 4]/2) * height     #右下角y
            cls = labels[:, 0]
            vis_gt_res = vis_gt(img, gt_boxes, cls,VOC_CLASSES)

        save_file_name = os.path.join(save_folder, os.path.basename(image_name))
        save_file_name_gt = save_file_name.replace('.jpg', "_gt.jpg")
        logger.info("Saving detection result in {}".format(save_file_name_gt))
        cv2.imwrite(save_file_name_gt, vis_gt_res)



def rodgers_visual(batch):

    num_img = batch['img'].shape[0]

    for i in range (num_img):
        img = batch['img'][i, :, :, :]
        img = img.numpy()
        img = np.transpose(img, [1, 2, 0])

        save_name = "batch-img" + str(i) + "_orgin.png"
        plt.imsave(save_name, img)

        batch_idx = batch['batch_idx'].tolist()
        # batch_idx = batch_idx.int()
        # gt_idx = batch_idx.index(i)

        gt_idx = [m for m,n in enumerate(batch_idx) if n == i]
        cls = batch['cls'][gt_idx]
        labels = batch['bboxes'][gt_idx]

        height, width = img.shape[:2]

        # img = img.astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)

        if labels.size == 0:
            vis_gt_res = img
        else:
            gt_boxes = (labels.numpy()).copy()
            gt_boxes[:, 0] = (labels[:, 0] - labels[:, 2]/2) * width      #左上角x
            gt_boxes[:, 1] = (labels[:, 1] - labels[:, 3]/2) * height     #左上角y
            gt_boxes[:, 2] = (labels[:, 0] + labels[:, 2]/2) * width      #右下角x
            gt_boxes[:, 3] = (labels[:, 1] + labels[:, 3]/2) * height     #右下角y
            vis_gt_res = vis_gt(img, gt_boxes, cls, CNRDD_CLASSES)

        save_folder = "./"

        save_file_name = os.path.join(save_folder, os.path.basename(save_name))
        save_file_name_gt = save_file_name.replace('_orgin.png', "_gt.jpg")
        logger.info("Saving detection result in {}".format(save_file_name_gt))
        cv2.imwrite(save_file_name_gt, vis_gt_res)

def rodgers_visual_img_and_mask(batch):

    num_img = batch['img'].shape[0]

    for i in range (num_img):
        img = batch['img'][i, :, :, :]
        img = img.numpy()
        img = np.transpose(img, [1, 2, 0])

        save_name = "batch-img" + str(i) + "_orgin.png"
        plt.imsave(save_name, img)

        batch_idx = batch['batch_idx'].tolist()
        # batch_idx = batch_idx.int()
        # gt_idx = batch_idx.index(i)

        gt_idx = [m for m,n in enumerate(batch_idx) if n == i]
        cls = batch['cls'][gt_idx]
        labels = batch['bboxes'][gt_idx]

        height, width = img.shape[:2]

        # img = img.astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)

        if labels.size == 0:
            vis_gt_res = img
        else:
            gt_boxes = (labels.numpy()).copy()
            gt_boxes[:, 0] = (labels[:, 0] - labels[:, 2]/2) * width      #左上角x
            gt_boxes[:, 1] = (labels[:, 1] - labels[:, 3]/2) * height     #左上角y
            gt_boxes[:, 2] = (labels[:, 0] + labels[:, 2]/2) * width      #右下角x
            gt_boxes[:, 3] = (labels[:, 1] + labels[:, 3]/2) * height     #右下角y
            vis_gt_res = vis_gt(img, gt_boxes, cls, CNRDD_CLASSES)

        save_folder = "./"

        save_file_name = os.path.join(save_folder, os.path.basename(save_name))
        save_file_name_gt = save_file_name.replace('_orgin.png', "_gt.jpg")
        # logger.info("Saving detection result in {}".format(save_file_name_gt))
        cv2.imwrite(save_file_name_gt, vis_gt_res)

    for i in range (num_img):
        img = batch['img_mask'][i, :, :, :]
        img = img.numpy()
        img = np.transpose(img, [1, 2, 0])

        batch_idx = batch['batch_idx'].tolist()
        # batch_idx = batch_idx.int()
        # gt_idx = batch_idx.index(i)

        gt_idx = [m for m,n in enumerate(batch_idx) if n == i]
        cls = batch['cls'][gt_idx]
        labels = batch['bboxes'][gt_idx]

        height, width = img.shape[:2]

        # img = img.astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)

        if labels.size == 0:
            vis_gt_res = img
        else:
            gt_boxes = (labels.numpy()).copy()
            gt_boxes[:, 0] = (labels[:, 0] - labels[:, 2]/2) * width      #左上角x
            gt_boxes[:, 1] = (labels[:, 1] - labels[:, 3]/2) * height     #左上角y
            gt_boxes[:, 2] = (labels[:, 0] + labels[:, 2]/2) * width      #右下角x
            gt_boxes[:, 3] = (labels[:, 1] + labels[:, 3]/2) * height     #右下角y
            vis_gt_res = vis_gt(img, gt_boxes, cls, CNRDD_CLASSES)

        save_folder = "./"
        save_name = "batch-img" + str(i) + "_mask.png"
        save_file_name = os.path.join(save_folder, os.path.basename(save_name))
        # logger.info("Saving detection result in {}".format(save_file_name_gt))
        cv2.imwrite(save_file_name, vis_gt_res)

        # save_name = "batch-img" + str(i) + "_mask.png"
        # plt.imsave(save_name, img)

def rodgers_visual_mask(batch_mask):

    num_img = batch_mask.shape[0]

    for i in range (num_img):
        img = batch_mask[i, :, :, :]
        img = img.numpy()
        img = np.transpose(img, [1, 2, 0])

        save_name = "batch-img" + str(i) + "_mask1.png"
        # plt.imsave(save_name, img)
        cv2.imwrite(save_name, img)

def rodgers_visual_mask_pred(mask_img):

    num_img = len(mask_img)

    for i in range (num_img):
        img = mask_img[i][0, :, :, :]
        img = img.cpu().numpy()
        img = img * 255
        img = np.transpose(img, [1, 2, 0])

        save_name = "mask_" + str(i) + "_pred.png"
        # plt.imsave(save_name, img)
        cv2.imwrite(save_name, img)



if __name__ == "__main__":
    args = make_parser().parse_args()       #解析参数

    main(args)
