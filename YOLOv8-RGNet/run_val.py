# -*- coding:utf-8 -*-

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO(r'./best.pt')
    model = YOLO(r'F:/博士研究生/paper/5-采用区域引导的道路损伤目标检测/实验记录/shiyan145/best.pt')

    # Validate the model
    model.val(
        val=True,  # (bool) validate/test during training
        # data=r'CNRDD.yaml',
        data=r'RDD2020.yaml',

        split='val',  # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
        batch=16,  # (int) number of images per batch (-1 for AutoBatch)
        imgsz=640,  # (int) size of input images as integer or w,h
        device='',  # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers=8,  # (int) number of worker threads for data loading (per RANK if DDP)
        save_json=False,  # (bool) save results to JSON file
        save_hybrid=False,  # (bool) save hybrid version of labels (labels + additional predictions)
        conf=0.001,  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        iou=0.5,  # (float) intersection over union (IoU) threshold for NMS
        project='runs/val',  # (str, optional) project name
        name='exp',  # (str, optional) experiment name, results saved to 'project/name' directory
        max_det=300,  # (int) maximum number of detections per image
        half=False,  # (bool) use half precision (FP16)
        dnn=False,  # (bool) use OpenCV DNN for ONNX inference
        plots=True,  # (bool) save plots during train/val
    )

