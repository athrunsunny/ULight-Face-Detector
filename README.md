# ULight-Face-Detector

Model size under 500KB

### Model size comparison
- Comparison of several open source lightweight face detection models:

Model|model file size（MB）
------|--------
Official Retinaface-Mobilenet-0.25 (Mxnet) | 1.68
Ultra-Light-Fast-Generic-Face-Detector-1MB(version-slim)| **1.04**
mobile_lite(our)| **0.5** 

## Generate VOC format training data set and training process

Download the wideface official website dataset or download the training set I provided and extract it into the ./data folder:

   The clean widerface data pack after filtering out the 10px*10px small face: [Baidu cloud disk (extraction code: cbiu)](https://pan.baidu.com/s/1MR0ZOKHUP_ArILjbAn03sw) 、[Google Drive](https://drive.google.com/open?id=1OBY-Pk5hkcVBX1dRBOeLI4e4OCvqJRnH )


## How to start

### install requirements
```Shell
pip install -r requirements.txt
```

### train
```Shell
python train.py --batch-size 16 --epochs 300 --weights '' --optimizer SGD --imgsz 320 (or 640)
```

#### prepare datasets

open data/widerface.yaml change train & val path 

### detect
```Shell
python detect.py --weights mobile_lite.pth 
```

### export
```Shell
python export.py --weights mobile_lite.pth 
```

## Pretrained model

Pretrained model: [Baidu cloud disk (extraction code: xagi)](https://pan.baidu.com/s/1IE92DusuINQC8RZ9NZ11QQ)
