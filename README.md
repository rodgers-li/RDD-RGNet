A Method of Road Damage Detection for Complex Background Images Based on Region Guidance Network
=
The overall structure of RDD-RGNet. It consists of five parts: backbone, neck, head, First-RG and Second-RG. First-RG fuses the extracted region information after spatial
attention with the feature extracted from Layer1 in backbone. Second-RG uses the RFF modules to efficiently fuse the three region-enhanced feature maps in backbone with the
three outputs of neck.
<div align="center">
  <img src="https://github.com/rodgers-li/RDD-RGNet/blob/main/overall.jpg">
</div>

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
