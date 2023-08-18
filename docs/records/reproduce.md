
# Unable to Reproduce

The implementation of this warehouse is based on the ultralytics/yolov5 warehouse, but from the implementation results, neither the training results of this warehouse nor ultralytics/yolov5 can reproduce the best model effect.

## ultralytics/yolov5

The training was conducted on the official Docker container, and the results of training from scratch are as follows,

```text
root@13cc41d4d571:/usr/src/app# python -m torch.distributed.run --nproc_per_node 4 --master_port 43212 train.py --data coco.yaml --weights "" --cfg yolov5s.yaml --img 640 --device 0,1,2,3
train: weights=, cfg=yolov5s.yaml, data=coco.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=100, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=0,1,2,3, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
remote: Enumerating objects: 24, done.
remote: Counting objects: 100% (24/24), done.
remote: Compressing objects: 100% (19/19), done.
remote: Total 24 (delta 8), reused 13 (delta 5), pack-reused 0
Unpacking objects: 100% (24/24), done.
From https://github.com/ultralytics/yolov5
   df48c20..dd10481  master     -> origin/master
github: ⚠️ YOLOv5 is out of date by 3 commits. Use 'git pull' or 'git clone https://github.com/ultralytics/yolov5' to update.
YOLOv5 🚀 v7.0-207-gdf48c20 Python-3.10.9 torch-2.0.0 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
                                                      CUDA:1 (NVIDIA GeForce RTX 3090, 24268MiB)
                                                      CUDA:2 (NVIDIA GeForce RTX 3090, 24268MiB)
                                                      CUDA:3 (NVIDIA GeForce RTX 3090, 24268MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1    229245  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
YOLOv5s summary: 214 layers, 7235389 parameters, 7235389 gradients, 16.6 GFLOPs

AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning /data/sde/coco/coco/train2017... 26573 images, 254 backgrounds, 0 corrupt:  23%|██▎       | 26827/118287 [00:08<00:31, 2939.98ittrain: Scanning /data/sde/coco/coco/train2017... 26867 images, 255 backgrounds, 0 corrutrain: Scanning /data/sde/coco/coco/train2017... 27167 images, 258 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 27455 images, 260 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 27742 images, 263 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 28029 images, 264 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 28311 images, 268 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 28612 images, 273 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 28939 images, 274 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 29243 images, 278 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 29552 images, 278 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 29856 images, 278 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 30172 images, 284 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 30484 images, 286 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 30795 images, 287 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 31104 images, 289 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 31411 images, 292 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 31790 images, 294 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 32109 images, 295 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 32424 images, 300 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 32741 images, 301 backgroundtrain: Scanning /data/sde/coco/coco/train2017... 33057 images, 304 backgroundtrain: Scanning /data/sde/coco/cotrain: Scanning /data/sde/coco/coco/train2017... 33687 images, 308 backgrounds, 0 corrupt:  train: Scanning /data/sde/coco/coco/train2017... 117266 images, 1021 backgrounds, 0 corrupt: 100%|██████████| 118287/118287 [00:38<00:00, 3068.13it/s]
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000099844.jpg: 2 duplicate labels removed
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000201706.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000214087.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000522365.jpg: 1 duplicate labels removed
train: New cache created: /data/sde/coco/coco/train2017.cache
val: Scanning /data/sde/coco/coco/val2017... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 [00:01<00:00, 3077.26it/s]
val: New cache created: /data/sde/coco/coco/val2017.cache

AutoAnchor: 4.45 anchors/target, 0.995 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/exp/labels.jpg... 
Image sizes 640 train, 640 val
Using 16 dataloader workers
Logging results to runs/train/exp
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/99      1.02G    0.08803    0.08695     0.0843         30        640: 100%|██████████| 7393/7393 [10:38<00:00, 11.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 [00:53<00:00, 11.59it/s]
                   all       5000      36335    0.00419      0.137    0.00819    0.00303
...
...
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      99/99      1.04G    0.04297    0.06723    0.02141         46        640: 100%|██████████| 7393/7393 [09:21<00:00, 13.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 [00:36<00:00, 17.00it/s]
                   all       5000      36335      0.633      0.496      0.532      0.343

100 epochs completed in 16.826 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.8MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.8MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
YOLOv5s summary: 157 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 [00:51<00:00, 12.17it/s]
                   all       5000      36335      0.631       0.49      0.528      0.343
...
...
Installing collected packages: pycocotools
Successfully installed pycocotools-2.0.7

requirements: AutoUpdate success ✅ 3.9s, installed 1 package: ['pycocotools>=2.0.6']
requirements: ⚠️ Restart runtime or rerun command for updates to take effect

loading annotations into memory...
Done (t=0.59s)
creating index...
index created!
Loading and preparing results...
DONE (t=8.71s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=94.99s).
Accumulating evaluation results...
DONE (t=20.84s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.536
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.199
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.397
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.698
Results saved to runs/train/exp
```

By retraining based on the weights obtained from scratch, the accuracy can be surprisingly discovered,

```text
root@13cc41d4d571:/usr/src/app# python -m torch.distributed.run --nproc_per_node 4 --master_port 43212 train.py --data coco.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3
train: weights=yolov5s.pt, cfg=, data=coco.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=100, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=0,1,2,3, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: ⚠️ YOLOv5 is out of date by 3 commits. Use 'git pull' or 'git clone https://github.com/ultralytics/yolov5' to update.
YOLOv5 🚀 v7.0-207-gdf48c20 Python-3.10.9 torch-2.0.0 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
                                                      CUDA:1 (NVIDIA GeForce RTX 3090, 24268MiB)
                                                      CUDA:2 (NVIDIA GeForce RTX 3090, 24268MiB)
                                                      CUDA:3 (NVIDIA GeForce RTX 3090, 24268MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1    229245  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7235389 parameters, 7235389 gradients, 16.6 GFLOPs

Transferred 349/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning /data/sde/coco/coco/train2017.cache... 117266 images, 1021 backgrounds, 0 corrupt: 100%|██████████| 118287/118287 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000099844.jpg: 2 duplicate labels removed
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000201706.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000214087.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/sde/coco/coco/images/train2017/000000522365.jpg: 1 duplicate labels removed
val: Scanning /data/sde/coco/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 [00:00<?, ?it/s]

AutoAnchor: 4.45 anchors/target, 0.995 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/exp3/labels.jpg... 
Image sizes 640 train, 640 val
Using 16 dataloader workers
Logging results to runs/train/exp3
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/99      1.02G    0.04338    0.06869    0.02437         30        640: 100%|██████████| 7393/7393 [10:25<00:00, 11.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 [00:39<00:00, 15.64it/s]
                   all       5000      36335      0.628      0.484      0.519      0.328
...
...
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      99/99      1.03G    0.04154    0.06544    0.01943         46        640: 100%|██████████| 7393/7393 [09:02<00:00, 13.62it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 [00:34<00:00, 18.14it/s]
                   all       5000      36335      0.663      0.509      0.557      0.365

100 epochs completed in 16.200 hours.
Optimizer stripped from runs/train/exp3/weights/last.pt, 14.8MB
Optimizer stripped from runs/train/exp3/weights/best.pt, 14.8MB

Validating runs/train/exp3/weights/best.pt...
Fusing layers... 
Model summary: 157 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 [00:47<00:00, 13.27it/s]
                   all       5000      36335      0.672      0.501      0.553      0.365
...
...
Evaluating pycocotools mAP... saving runs/train/exp3/_predictions.json...
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
Loading and preparing results...
DONE (t=7.90s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=89.00s).
Accumulating evaluation results...
DONE (t=19.48s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.717
Results saved to runs/train/exp3
```

ultralytics/yolov5 trained a total of 300 epochs and achieved the best results. After two training sessions of 100 epochs each, the results can also be very close to the original training results, indicating that more training sessions are necessary for YOLOv5.

## Mine

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