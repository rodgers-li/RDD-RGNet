A Method of Road Damage Detection for Complex Background Images Based on Region Guidance Network
=
Computer vision techniques are the most commonly used methods in automatic road damage detection. However, road damage detection in front-view images is a challenging task due to the complex background. In order to localize the damages more accurately in complex background images, we propose a two-stage region guidance method. In the first-stage region guidance network, a region segmentation task is added to the object detection network using path and weight sharing. The segmented region information is aggregated with the features in the backbone through spatial attention to guide the backbone to enhance the feature extraction in damaged regions. In the second-stage region guidance network, the region-enhanced features in the backbone are efficiently fused with the outputs of neck through the region feature fusion modules to further guide the model for precise localization. We use YOLOv8 as the baseline and propose a road damage detection network based on region guidance, called RDD-RGNet.

## Get start

### Training
```Python
python run_train.py
```

### testing
```Python
python run_val.py
```

### prediction
```Python
python run_detect.py
```

## Acknowledgment
This implementation is bulit upon YOLOv8.
