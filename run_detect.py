from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('./best.pt') 
    model.predict(
        source=r'/var/lib/docker/rdd2020/val_image',

        save=True,  # save predict results
        imgsz=640,  # (int) size of input images as integer or w,h
        conf=0.25,  # object confidence threshold for detection (default 0.25 predict, 0.001 val)
        iou=0.5,  # # intersection over union (IoU) threshold for NMS
        show=False,  # show results if possible
        project='runs/predict',  # (str, optional) project name
        name='exp',  # (str, optional) experiment name, results saved to 'project/name' directory
        save_txt=False,  # save results as .txt file
        save_conf=True,  # save results with confidence scores
        save_crop=False,  # save cropped images with results
        show_labels=True,  # show object labels in plots
        show_conf=True,  # show object confidence scores in plots
        vid_stride=1,  # video frame-rate stride
        line_width=3,  # bounding box thickness (pixels)
        visualize=False,  # visualize model features
        augment=False,  # apply image augmentation to prediction sources
        agnostic_nms=False,  # class-agnostic NMS
        retina_masks=False,  # use high-resolution segmentation masks
        boxes=True,  # Show boxes in segmentation predictions
    )

