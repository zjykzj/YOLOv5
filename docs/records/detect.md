# Detect

* COCO
  * yolov5x
  * yolov5l
  * yolov5m
  * yolov5n
  * yolov5s
  * yolov3
* VOC
    * yolov5s
    * yolov3
    * yolov3-tiny

## YOLOv5x with COCO

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 36232 train.py  --data coco.yaml --weights "" --cfg yolov5x.yaml --img 640 --device 0,1,2,3
...
...
100 epochs completed in 136.361 hours.                                                                                                                                                                                                                                            
Optimizer stripped from runs/train/exp17/weights/last.pt, 174.1MB                                                                                                                                                                                                                 
Optimizer stripped from runs/train/exp17/weights/best.pt, 174.1MB                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                  
Validating runs/train/exp17/weights/best.pt...                                                                                                                                                                                                                                    
Fusing layers...                                                                                                                                                                                                                                                                  
YOLOv5x summary: 322 layers, 86705005 parameters, 0 gradients, 205.5 GFLOPs                                                                                                                                                                                                       
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 03:13                                                                                                                                                          
                   all       5000      36335      0.729      0.599      0.659      0.479      
...
...
Evaluating pycocotools mAP... saving runs/train/exp17/_predictions.json...
loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=6.23s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=53.64s).
Accumulating evaluation results...
DONE (t=12.22s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.665
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.661
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.717
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.813
Results saved to runs/train/exp17
```

```text
$ python val.py --weights runs/train/exp17/weights/best.pt --data coco.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/coco.yaml, weights=['runs/train/exp17/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-8 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5x summary: 322 layers, 86705005 parameters, 0 gradients, 205.5 GFLOPs
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 03:03
                   all       5000      36335      0.727      0.603      0.661      0.478
Speed: 0.2ms pre-process, 17.0ms inference, 2.5ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp15/best_predictions.json...
loading annotations into memory...
Done (t=0.40s)
creating index...
index created!
Loading and preparing results...
DONE (t=4.32s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=46.90s).
Accumulating evaluation results...
DONE (t=10.36s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.667
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.532
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.654
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.710
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.808
Results saved to runs/val/exp15
```

## YOLOv5l with COCO

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 53122 train.py --data coco.yaml --weights "" --cfg yolov5l.yaml --img 640 --device 0,1,2,3
...
...
100 epochs completed in 114.272 hours.                              
Optimizer stripped from runs/train/exp15/weights/last.pt, 93.6MB                                                                         
Optimizer stripped from runs/train/exp15/weights/best.pt, 93.6MB                                                                         

Validating runs/train/exp15/weights/best.pt...                      
Fusing layers...                                                    
YOLOv5l summary: 267 layers, 46533693 parameters, 0 gradients, 109.0 GFLOPs                                                              
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 14:36                                                                                                                                                          
                   all       5000      36335      0.704      0.589      0.642       0.46           
...
...
Evaluating pycocotools mAP... saving runs/train/exp15/_predictions.json...
loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=5.97s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=53.13s).
Accumulating evaluation results...
DONE (t=12.24s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.650
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.505
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.360
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.647
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.705
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.791
Results saved to runs/train/exp15
```

```shell
$ python val.py --weights runs/train/exp15/weights/best.pt --data coco.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/coco.yaml, weights=['runs/train/exp15/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-8 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5l summary: 267 layers, 46533693 parameters, 0 gradients, 109.0 GFLOPs
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 02:21
                   all       5000      36335      0.704      0.592      0.645      0.459
Speed: 0.2ms pre-process, 9.3ms inference, 2.2ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp14/best_predictions.json...
loading annotations into memory...
Done (t=0.41s)
creating index...
index created!
Loading and preparing results...
DONE (t=5.21s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=49.23s).
Accumulating evaluation results...
DONE (t=11.54s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.652
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.595
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.360
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.639
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.695
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.786
Results saved to runs/val/exp14
```

## YOLOv5m with COCO

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 56122 train.py --data coco.yaml --weights "" --cfg yolov5m.yaml --img 640 --device 4,5,6,7
...
...
100 epochs completed in 58.639 hours.                                                                                                                                                                                                                                             
Optimizer stripped from runs/train/exp16/weights/last.pt, 42.7MB                                                                                                                                                                                                                  
Optimizer stripped from runs/train/exp16/weights/best.pt, 42.7MB                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                  
Validating runs/train/exp16/weights/best.pt...                                                                                                                                                                                                                                    
Fusing layers...                                                                                                                                                                                                                                                                  
YOLOv5m summary: 212 layers, 21172173 parameters, 0 gradients, 48.9 GFLOPs                                                                                                                                                                                                        
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 04:34                                                                                                                                                          
                   all       5000      36335      0.681      0.565      0.609      0.425         
...
...
Evaluating pycocotools mAP... saving runs/train/exp16/_predictions.json...
loading annotations into memory...
Done (t=1.29s)
creating index...
index created!
Loading and preparing results...
DONE (t=6.92s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=54.58s).
Accumulating evaluation results...
DONE (t=17.92s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.617
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.676
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.774
Results saved to runs/train/exp16
```

```shell
$ python val.py --weights runs/train/exp16/weights/best.pt --data coco.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/coco.yaml, weights=['runs/train/exp16/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-8 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5m summary: 212 layers, 21172173 parameters, 0 gradients, 48.9 GFLOPs
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:50
                   all       5000      36335      0.681      0.573      0.613      0.426
Speed: 0.1ms pre-process, 3.7ms inference, 1.2ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp16/best_predictions.json...
	loading annotations into memory...
Done (t=0.78s)
creating index...
index created!
Loading and preparing results...
DONE (t=4.90s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=51.29s).
Accumulating evaluation results...
DONE (t=12.16s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.621
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.668
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.766
Results saved to runs/val/exp16
```

## YOLOv5n with COCO

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 56122 train.py --data coco.yaml --weights "" --cfg yolov5n.yaml --img 640 --device 4,5,6,7
...
...
100 epochs completed in 20.924 hours.                               
Optimizer stripped from runs/train/exp14/weights/last.pt, 4.0MB                                                                          
Optimizer stripped from runs/train/exp14/weights/best.pt, 4.0MB                                                                          

Validating runs/train/exp14/weights/best.pt...                      
Fusing layers...                                                    
YOLOv5n summary: 157 layers, 1867405 parameters, 0 gradients, 4.5 GFLOPs                                                                 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 00:56                                                                                                                                                          
                   all       5000      36335      0.543      0.374      0.401       0.24  
...
...
Evaluating pycocotools mAP... saving runs/train/exp14/_predictions.json...
loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=8.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=61.84s).
Accumulating evaluation results...
DONE (t=16.10s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.279
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.235
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619
Results saved to runs/train/exp14
```

```shell
$ python val.py --weights runs/train/exp14/weights/best.pt --data coco.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/coco.yaml, weights=['runs/train/exp14/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5n summary: 157 layers, 1867405 parameters, 0 gradients, 4.5 GFLOPs
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:44
                   all       5000      36335      0.537      0.384      0.406      0.241
Speed: 0.1ms pre-process, 1.3ms inference, 2.1ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp8/best_predictions.json...
loading annotations into memory...
Done (t=0.38s)
creating index...
index created!
Loading and preparing results...
DONE (t=6.26s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=53.94s).
Accumulating evaluation results...
DONE (t=14.59s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.121
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.278
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.236
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.408
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
Results saved to runs/val/exp8
```

## YOLOv5s with COCO

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 53122 train.py --data coco.yaml --weights "" --cfg yolov5s.yaml --img 640 --device 0,1,2,3
...
...
100 epochs completed in 36.592 hours.                                                                                                                                                                                                                                             
Optimizer stripped from runs/train/exp10/weights/last.pt, 14.8MB                                                                                                                                                                                                                  
Optimizer stripped from runs/train/exp10/weights/best.pt, 14.8MB                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                  
Validating runs/train/exp10/weights/best.pt...                                                                                                                                                                                                                                    
Fusing layers...                                                                                                                                                                                                                                                                  
YOLOv5s summary: 157 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs                                                                                                                                                                                                         
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 2/625 00:01WARNING ⚠️ NMS time limit 0.900s exceeded                                                                                                                    
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 00:56                                                                                                                                                          
                   all       5000      36335      0.634      0.487      0.525      0.342
...
...
Evaluating pycocotools mAP... saving runs/train/exp10/_predictions.json...
loading annotations into memory...
Done (t=1.74s)
creating index...
index created!
Loading and preparing results...
DONE (t=9.65s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=59.19s).
Accumulating evaluation results...
DONE (t=14.55s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.353
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.709
Results saved to runs/train/exp10
```

```shell
$ python val.py --weights runs/train/exp10/weights/best.pt --data coco.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/coco.yaml, weights=['runs/train/exp10/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5s summary: 157 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:43
                   all       5000      36335      0.639      0.492       0.53      0.342
Speed: 0.1ms pre-process, 2.1ms inference, 1.4ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp4/best_predictions.json...
loading annotations into memory...
Done (t=0.38s)
creating index...
index created!
Loading and preparing results...
DONE (t=6.32s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=48.44s).
Accumulating evaluation results...
DONE (t=12.19s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.693
Results saved to runs/val/exp4
```

## YOLOv5s with VOC

```shell
python -m torch.distributed.run --nproc_per_node 4 --master_port 36232 train.py  --data VOC.yaml --weights "" --cfg yolov5s.yaml --img 640 --device 0,1,2,3
...
...
100 epochs completed in 22.370 hours.                                                                                                                                                                                                                                             
Optimizer stripped from runs/train/exp4/weights/last.pt, 14.4MB                                                                                                                                                                                                                   
Optimizer stripped from runs/train/exp4/weights/best.pt, 14.4MB                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                  
Validating runs/train/exp4/weights/best.pt...                                                                                                                                                                                                                                     
Fusing layers...                                                                                                                                                                                                                                                                  
YOLOv5s summary: 157 layers, 7064065 parameters, 0 gradients, 15.9 GFLOPs                                                                                                                                                                                                         
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 619/619 10:31                                                                                                                                                          
                   all       4952      12032      0.728      0.683      0.738      0.468                                                                                                                                                                                          
             aeroplane       4952        285      0.845      0.754       0.83      0.502                                                                                                                                                                                          
               bicycle       4952        337      0.843      0.777      0.851      0.554                                                                                                                                                                                          
                  bird       4952        459      0.705      0.621      0.679      0.392                                                                                                                                                                                          
                  boat       4952        263      0.615      0.612      0.635      0.348                                                                                                                                                                                          
                bottle       4952        469      0.664      0.636      0.672      0.399                                                                                                                                                                                          
                   bus       4952        213      0.796      0.768      0.819      0.636                                                                                                                                                                                          
                   car       4952       1201      0.806      0.849      0.894      0.648                                                                                                                                                                                          
                   cat       4952        358      0.805      0.651      0.769      0.507                                                                                                                                                                                          
                 chair       4952        756      0.606      0.534      0.577       0.33                                                                                                                                                                                          
                   cow       4952        244      0.663      0.692      0.739        0.5                                                                                                                                                                                          
           diningtable       4952        206      0.702      0.646      0.687        0.4                                                                                                                                                                                          
                   dog       4952        489      0.756      0.583      0.728      0.447                                                                                                                                                                                          
                 horse       4952        348      0.784      0.776       0.81      0.511                                                                                                                                                                                          
             motorbike       4952        325      0.836      0.714      0.825      0.511                                                                                                                                                                                          
                person       4952       4528      0.809      0.766      0.833      0.498                                                                                                                                                                                          
           pottedplant       4952        480      0.602      0.456      0.476      0.215                                                                                                                                                                                          
                 sheep       4952        242      0.602       0.76      0.737      0.504                                                                                                                                                                                          
                  sofa       4952        239      0.627      0.607      0.649       0.44                                                                                                                                                                                          
                 train       4952        282      0.812      0.748      0.803      0.513                                                                                                                                                                                          
             tvmonitor       4952        308      0.684      0.718      0.754      0.502   
```

```shell
$ python val.py --weights runs/train/exp4/weights/best.pt --data VOC.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/VOC.yaml, weights=['runs/train/exp4/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5s summary: 157 layers, 7064065 parameters, 0 gradients, 15.9 GFLOPs
val: Scanning /home/zj/pp/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 00:27
                   all       4952      12032      0.729      0.682      0.738      0.468
             aeroplane       4952        285      0.843      0.754      0.829      0.503
               bicycle       4952        337       0.84      0.777       0.85      0.555
                  bird       4952        459      0.709      0.621      0.678      0.394
                  boat       4952        263      0.616      0.612      0.635      0.345
                bottle       4952        469      0.664      0.633      0.674      0.401
                   bus       4952        213      0.791      0.765      0.819      0.636
                   car       4952       1201      0.808      0.846      0.894      0.649
                   cat       4952        358      0.806      0.648      0.769      0.507
                 chair       4952        756      0.609       0.53      0.577       0.33
                   cow       4952        244      0.662       0.69      0.739      0.497
           diningtable       4952        206      0.703      0.645      0.688      0.401
                   dog       4952        489      0.752      0.579      0.724      0.444
                 horse       4952        348      0.788      0.779      0.812       0.51
             motorbike       4952        325      0.835      0.715      0.825      0.508
                person       4952       4528      0.812      0.765      0.834      0.498
           pottedplant       4952        480        0.6       0.45      0.477      0.216
                 sheep       4952        242      0.604       0.76      0.737      0.501
                  sofa       4952        239      0.625      0.607      0.647       0.44
                 train       4952        282      0.811      0.748      0.802      0.514
             tvmonitor       4952        308      0.693      0.718      0.754      0.505
Speed: 0.1ms pre-process, 1.4ms inference, 0.8ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val/exp3
```

## YOLOv3 with COCO

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 53122 train.py --data coco.yaml --weights "" --cfg yolov3.yaml --img 640 --device 0,1,2,3
...
...
100 epochs completed in 43.008 hours.                                          
Optimizer stripped from runs/train/exp13/weights/last.pt, 124.3MB                                                                                             
Optimizer stripped from runs/train/exp13/weights/best.pt, 124.3MB                                                                                             

Validating runs/train/exp13/weights/best.pt...                                 
Fusing layers...                                                               
yolov3 summary: 190 layers, 61922845 parameters, 0 gradients, 155.9 GFLOPs                                                                                    
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 00:57
                   all       5000      36335      0.713      0.574      0.626      0.433   
...
...
Evaluating pycocotools mAP... saving runs/train/exp13/_predictions.json...
loading annotations into memory...
Done (t=0.37s)
creating index...
index created!
Loading and preparing results...
DONE (t=5.64s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=46.40s).
Accumulating evaluation results...
DONE (t=10.36s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.436
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.633
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.475
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.276
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.675
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
Results saved to runs/train/exp13
```

```shell
$ python val.py --weights runs/train/exp13/weights/best.pt --data coco.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/coco.yaml, weights=['runs/train/exp13/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
yolov3 summary: 190 layers, 61922845 parameters, 0 gradients, 155.9 GFLOPs
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 01:05
                   all       5000      36335      0.717      0.575       0.63      0.433
Speed: 0.1ms pre-process, 7.0ms inference, 0.9ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp6/best_predictions.json...
loading annotations into memory...
Done (t=0.37s)
creating index...
index created!
Loading and preparing results...
DONE (t=4.82s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=45.73s).
Accumulating evaluation results...
DONE (t=10.17s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.436
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.637
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.272
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.767
Results saved to runs/val/exp6
```

## YOLOv3 with VOC

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 36232 train.py  --data VOC.yaml --weights "" --cfg yolov3.yaml --img 640 --device 0,1,2,3
...
...
100 epochs completed in 17.658 hours.
Optimizer stripped from runs/train/exp12/weights/last.pt, 123.6MB
Optimizer stripped from runs/train/exp12/weights/best.pt, 123.6MB

Validating runs/train/exp12/weights/best.pt...
Fusing layers... 
yolov3 summary: 190 layers, 61599745 parameters, 0 gradients, 154.9 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 619/619 03:37
                   all       4952      12032      0.794      0.763      0.819      0.569
             aeroplane       4952        285       0.88      0.835      0.897      0.604
               bicycle       4952        337       0.91      0.816      0.908      0.639
                  bird       4952        459      0.794      0.693      0.777      0.509
                  boat       4952        263      0.633      0.683        0.7       0.42
                bottle       4952        469      0.729      0.704      0.744      0.499
                   bus       4952        213       0.84       0.85      0.904      0.738
                   car       4952       1201       0.82      0.898       0.93      0.703
                   cat       4952        358       0.87      0.793      0.866      0.626
                 chair       4952        756      0.665      0.638      0.675      0.441
                   cow       4952        244      0.735      0.842      0.859      0.624
           diningtable       4952        206      0.732      0.718      0.731      0.483
                   dog       4952        489      0.856      0.703      0.847      0.605
                 horse       4952        348       0.88      0.842      0.909      0.666
             motorbike       4952        325      0.892      0.803      0.883      0.592
                person       4952       4528      0.857      0.797      0.885      0.584
           pottedplant       4952        480      0.665      0.506      0.557      0.293
                 sheep       4952        242      0.738      0.813      0.837      0.602
                  sofa       4952        239      0.717      0.695      0.776      0.556
                 train       4952        282      0.855      0.835      0.875      0.604
             tvmonitor       4952        308      0.807      0.792      0.825      0.593
Results saved to runs/train/exp12
```

```shell
$ python val.py --weights runs/train/exp12/weights/best.pt --data VOC.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/VOC.yaml, weights=['runs/train/exp12/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
yolov3 summary: 190 layers, 61599745 parameters, 0 gradients, 154.9 GFLOPs
val: Scanning /home/zj/pp/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 00:51
                   all       4952      12032      0.794      0.762      0.819      0.569
             aeroplane       4952        285      0.881      0.835      0.897      0.606
               bicycle       4952        337       0.91      0.816      0.908      0.639
                  bird       4952        459      0.799      0.695      0.777       0.51
                  boat       4952        263      0.633      0.673       0.69      0.416
                bottle       4952        469       0.73      0.706      0.744      0.498
                   bus       4952        213      0.842      0.849      0.905      0.739
                   car       4952       1201      0.819      0.898       0.93      0.704
                   cat       4952        358      0.871      0.793      0.866      0.625
                 chair       4952        756      0.665      0.638      0.675       0.44
                   cow       4952        244      0.735      0.841      0.859      0.626
           diningtable       4952        206      0.728      0.713      0.731      0.483
                   dog       4952        489      0.858      0.703      0.847      0.602
                 horse       4952        348      0.882      0.845      0.909      0.667
             motorbike       4952        325       0.89        0.8      0.883      0.594
                person       4952       4528      0.857      0.797      0.884      0.584
           pottedplant       4952        480      0.665      0.504      0.558      0.294
                 sheep       4952        242       0.74      0.813      0.837      0.601
                  sofa       4952        239      0.719       0.69      0.777      0.557
                 train       4952        282      0.852      0.835      0.874      0.602
             tvmonitor       4952        308      0.807      0.792      0.825      0.593
Speed: 0.1ms pre-process, 6.3ms inference, 0.7ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val/exp7
```

## YOLOv3-Tiny with VOC

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 36232 train.py  --data VOC.yaml --weights "" --cfg yolov3-tiny.yaml --img 640 --device 0,1,2,3
...
...
100 epochs completed in 6.260 hours.
Optimizer stripped from runs/train/exp11/weights/last.pt, 17.5MB
Optimizer stripped from runs/train/exp11/weights/best.pt, 17.5MB

Validating runs/train/exp11/weights/best.pt...
Fusing layers... 
yolov3-tiny summary: 38 layers, 8710582 parameters, 0 gradients, 13.0 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 619/619 01:06
                   all       4952      12032      0.563      0.551      0.542      0.253
             aeroplane       4952        285      0.508      0.611       0.55      0.211
               bicycle       4952        337      0.707      0.656      0.717      0.362
                  bird       4952        459      0.593      0.393      0.477      0.218
                  boat       4952        263      0.484      0.479      0.456      0.188
                bottle       4952        469       0.54      0.447      0.455      0.195
                   bus       4952        213      0.504      0.638      0.543      0.268
                   car       4952       1201      0.572       0.77      0.707      0.348
                   cat       4952        358      0.294      0.592      0.302     0.0934
                 chair       4952        756      0.584      0.426      0.466      0.233
                   cow       4952        244      0.586      0.615      0.639      0.357
           diningtable       4952        206      0.496      0.257      0.348      0.113
                   dog       4952        489      0.472      0.485      0.435      0.172
                 horse       4952        348      0.661      0.641      0.656      0.292
             motorbike       4952        325      0.672      0.668      0.697      0.328
                person       4952       4528      0.713      0.681      0.729      0.349
           pottedplant       4952        480       0.51      0.326      0.357      0.136
                 sheep       4952        242       0.61      0.669      0.667      0.396
                  sofa       4952        239      0.533      0.392      0.411      0.184
                 train       4952        282      0.528      0.652      0.562      0.228
             tvmonitor       4952        308      0.696      0.618      0.658      0.384
Results saved to runs/train/exp11
```

```shell
$ python val.py --weights runs/train/exp11/weights/best.pt --data VOC.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/VOC.yaml, weights=['runs/train/exp11/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
yolov3-tiny summary: 38 layers, 8710582 parameters, 0 gradients, 13.0 GFLOPs
val: Scanning /home/zj/pp/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 155/155 00:24
                   all       4952      12032      0.563      0.552      0.542      0.253
             aeroplane       4952        285      0.506      0.611       0.55       0.21
               bicycle       4952        337      0.707      0.659       0.72      0.364
                  bird       4952        459      0.604      0.397      0.478      0.218
                  boat       4952        263      0.487      0.475      0.457      0.189
                bottle       4952        469      0.537      0.447      0.451      0.193
                   bus       4952        213      0.504      0.643      0.543      0.268
                   car       4952       1201       0.57      0.768      0.706      0.349
                   cat       4952        358      0.292      0.592      0.302     0.0931
                 chair       4952        756      0.588      0.426      0.467      0.232
                   cow       4952        244      0.593      0.621      0.641      0.359
           diningtable       4952        206      0.495      0.257      0.348      0.113
                   dog       4952        489      0.466      0.483      0.433      0.171
                 horse       4952        348      0.658      0.641      0.658      0.295
             motorbike       4952        325      0.661      0.665      0.696      0.324
                person       4952       4528       0.71      0.682      0.729      0.348
           pottedplant       4952        480       0.51      0.327      0.359      0.136
                 sheep       4952        242      0.611      0.669      0.666      0.398
                  sofa       4952        239      0.536      0.402       0.41      0.184
                 train       4952        282      0.532      0.652       0.56      0.226
             tvmonitor       4952        308      0.699       0.62      0.663      0.386
Speed: 0.1ms pre-process, 1.1ms inference, 0.7ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val/exp5
```