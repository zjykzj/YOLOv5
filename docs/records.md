
# Records

## Train

### YOLOv5s with VOC

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 36232 train.py  --data VOC.yaml --weights "" --cfg yolov5s.yaml --img 640 --device 4,5,6,7
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

### YOLOv5s with COCO

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

### YOLOv5-Tiny with VOC

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

## Eval

### YOLOv5 with COCO

```text
$ python val.py --weights runs/train/exp10/weights/best.pt --data coco.yaml --img 640
val: data=/home/zj/pp/YOLOv5/configs/data/coco.yaml, weights=['runs/train/exp10/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5s summary: 157 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:46
                   all       5000      36335      0.639      0.492       0.53      0.342
Speed: 0.1ms pre-process, 2.2ms inference, 1.5ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp/best_predictions.json...
loading annotations into memory...
Done (t=0.41s)
creating index...
index created!
Loading and preparing results...
DONE (t=6.74s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=55.11s).
Accumulating evaluation results...
DONE (t=14.24s).
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
Results saved to runs/val/exp
```

## Predict

### YOLOv5 with COCO

```text
$ python detect.py --weights runs/train/exp10/weights/best.pt --source assets/coco/
detect: weights=['runs/train/exp10/weights/best.pt'], source=assets/coco/, data=configs/data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-1 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
YOLOv5s summary: 157 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
image 1/2 /home/zj/pp/YOLOv5/assets/coco/bus.jpg: 640x480 3 persons, 1 bus, 1 skateboard, 13.7ms
image 2/2 /home/zj/pp/YOLOv5/assets/coco/zidane.jpg: 384x640 2 persons, 3 ties, 12.9ms
Speed: 0.4ms pre-process, 13.3ms inference, 1.3ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp3
```