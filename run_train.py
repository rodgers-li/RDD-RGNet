# -*- coding:utf-8 -*-

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO(r'yolov8.yaml')  # 不使用预训练权重训练
    # model = YOLO(r'yolov8s.yaml').load("yolov8s.pt")  # 使用预训练权重训练
    model = YOLO(r'yolov8s-RG-cross-attention.yaml').load("yolov8s.pt")  # 使用预训练权重训练

    # Trainparameters ------------------------------------------------------------

    model.train(
        # data=r'CNRDD.yaml',
        data=r'RDD2020.yaml',

        epochs= 100 , # (int) number of epochs to train for
        patience= 100 , # (int) epochs to wait for no observable improvement for early stopping of training
        batch= 16 , # (int) number of images per batch (-1 for AutoBatch)
        imgsz= 640 , # (int) size of input images as integer or w,h
        save= True , # (bool) save train checkpoints and predict results
        save_period= -1, # (int) Save checkpoint every x epochs (disabled if < 1)
        cache= False , # (bool) True/ram, disk or False. Use cache for data loading
        device= '0' , # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers= 4 , # (int) number of worker threads for data loading (per RANK if DDP)
        project= 'runs/train', # (str, optional) project name
        name= 'exp' ,# (str, optional) experiment name, results saved to 'project/name' directory
        exist_ok= False , # (bool) whether to overwrite existing experiment
        pretrained= True , # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
        optimizer= 'SGD',  # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        verbose= True ,# (bool) whether to print verbose output
        seed= 0 , # (int) random seed for reproducibility
        deterministic= True , # (bool) whether to enable deterministic mode
        single_cls= False , # (bool) train multi-class data as single-class
        rect= False  , # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
        cos_lr= False , # (bool) use cosine learning rate scheduler
        close_mosaic= 0,  # (int) disable mosaic augmentation for final epochs
        resume= False , # (bool) resume training from last checkpoint
        amp= True,  # (bool) Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
        fraction= 1.0 , # (float) dataset fraction to train on (default is 1.0, all images in train set)
        profile= False,  # (bool) profile ONNX and TensorRT speeds during training for loggers
        conf=0.001,  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        iou=0.5,  # (float) intersection over union (IoU) threshold for NMS
        plots=True,  # (bool) save plots during train/val
        # Segmentation
        overlap_mask= True , # (bool) masks should overlap during training (segment train only)
        mask_ratio= 4, # (int) mask downsample ratio (segment train only)
        # Classification
        dropout= 0.0,  # (float) use dropout regularization (classify train only)
        # Hyperparameters ----------------------------------------------------------------------------------------------
        lr0=0.01,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        lrf=0.01,  # (float) final learning rate (lr0 * lrf)
        momentum=0.937,  # (float) SGD momentum/Adam beta1
        weight_decay=0.0005,  # (float) optimizer weight decay 5e-4
        warmup_epochs=3.0,  # (float) warmup epochs (fractions ok)
        warmup_momentum=0.8,  # (float) warmup initial momentum
        warmup_bias_lr=0.1,  # (float) warmup initial bias lr
        box=7.5,  # (float) box loss gain
        cls=0.5,  # (float) cls loss gain (scale with pixels)
        dfl=1.5,  # (float) dfl loss gain
        pose=12.0,  # (float) pose loss gain
        kobj=1.0,  # (float) keypoint obj loss gain
        label_smoothing=0.0,  # (float) label smoothing (fraction)
        nbs=64,  # (int) nominal batch size
        hsv_h=0.015,  # (float) image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # (float) image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # (float) image HSV-Value augmentation (fraction)
        degrees=0.0,  # (float) image rotation (+/- deg)
        translate=0.1,  # (float) image translation (+/- fraction)
        scale=0.5,  # (float) image scale (+/- gain)
        shear=0.0,  # (float) image shear (+/- deg)
        perspective=0.0,  # (float) image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # (float) image flip up-down (probability)
        fliplr=0.5,  # (float) image flip left-right (probability)
        mosaic=1.0,  # (float) image mosaic (probability)
        mixup=0.0,  # (float) image mixup (probability)
        copy_paste=0.0,  # (float) segment copy-paste (probability)
    )


