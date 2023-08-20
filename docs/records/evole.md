
# Evolve

## YOLOv5s with VOC

```shell
python evolve.py --batch 128 --weights "" --cfg yolov5s.yaml --data VOC.yaml --epochs 50 --img 512 --hyp hyp.scratch-med.yaml --evolve --device 5
...
...
Best results from row 266 of runs/evolve/exp3/evolve.csv:                                                                                                                                                          
            lr0: 0.0112                                                                                                                                                                                            
            lrf: 0.0828                                                                                                                                                                                            
       momentum: 0.98                                                                                                                                                                                              
   weight_decay: 0.00031                                                                                                                                                                                           
  warmup_epochs: 3.11                                                                                                                                                                                              
warmup_momentum: 0.88                                                                                                                                                                                              
 warmup_bias_lr: 0.0781                                                                                                                                                                                            
            box: 0.0319                                                                                                                                                                                            
            cls: 0.39                                                                                                                                                                                              
         cls_pw: 0.565                                                                                                                                                                                             
            obj: 0.967                                                                                                                                                                                             
         obj_pw: 1.11                                                                                                                                                                                              
          iou_t: 0.2                                                                                                                                                                                               
       anchor_t: 3.83                                                                                                                                                                                              
       fl_gamma: 0                                                                                                                                                                                                 
          hsv_h: 0.0202                                                                                                                                                                                            
          hsv_s: 0.75                                                                                                                                                                                              
          hsv_v: 0.625                                                                                                                                                                                             
        degrees: 0                                                                                                                                                                                                 
      translate: 0.106                                                                                                                                                                                             
          scale: 0.511                                                                                                                                                                                             
          shear: 0                                                                                                                                                                                                 
    perspective: 0                                                                                                                                                                                                 
         flipud: 0                                                                                       
         fliplr: 0.5                                                                                     
         mosaic: 0.84                                                                                    
          mixup: 0.0778                                                                                  
     copy_paste: 0                                                                                       
        anchors: 3.17                                                                                    
Saved runs/evolve/exp3/evolve.png                                                                        
Hyperparameter evolution finished 300 generations                                                        
Results saved to runs/evolve/exp3                                                                        
Usage example: $ python train.py --hyp runs/evolve/exp3/hyp_evolve.yaml  
```

```shell
python -m torch.distributed.run --nproc_per_node 4 --master_port 36232 train.py  --data VOC.yaml --weights "" --cfg yolov5s.yaml --img 640 --hyp runs/evolve/exp3/hyp_evolve.ya
ml --device 0,1,2,3                                                                                                                                                                                                
WARNING:__main__:                                                                                                                                                                                                  
*****************************************                                                                                                                                                                          
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.   
*****************************************                                                                                                                                                                          
train: weights=, cfg=yolov5s.yaml, data=VOC.yaml, hyp=runs/evolve/exp3/hyp_evolve.yaml, epochs=100, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=Fals
e, evolve=None, bucket=, cache=None, image_weights=False, device=0,1,2,3, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, c
os_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest                                      
github: skipping check (not a git repository), for updates see https://github.com/zjykzj/YOLOv5                                                                                                                    
YOLOv5 🚀 2023-8-8 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)                                                                                                                     
                                                    CUDA:1 (NVIDIA GeForce RTX 3090, 24268MiB)                                                                                                                     
                                                    CUDA:2 (NVIDIA GeForce RTX 3090, 24268MiB)                                                                                                                     
                                                    CUDA:3 (NVIDIA GeForce RTX 3090, 24268MiB)                                                                                                                     
                                                                                                                                                                                                                   
hyperparameters: lr0=0.01115, lrf=0.08284, momentum=0.98, weight_decay=0.00031, warmup_epochs=3.1106, warmup_momentum=0.87978, warmup_bias_lr=0.07814, box=0.0319, cls=0.39042, cls_pw=0.56542, obj=0.96735, obj_pw
=1.1099, iou_t=0.2, anchor_t=3.8327, fl_gamma=0.0, hsv_h=0.02015, hsv_s=0.7496, hsv_v=0.6246, degrees=0.0, translate=0.10601, scale=0.51105, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.84024, mi
xup=0.07779, copy_paste=0.0, anchors=3.1689                                                                                                                                                                        
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/                                                                                                                          
Overriding model.yaml nc=80 with nc=20                                                                                                                                                                             
Overriding model.yaml anchors with anchors=3.1689      
                                                                                                                                                                                                                   
                 from  n    params  module                                  arguments                                                                                                                              
  0                -1  1      3520  yolo.model.impl.common.Conv             [3, 32, 6, 2, 2]                                                                                                                       
  1                -1  1     18560  yolo.model.impl.common.Conv             [32, 64, 3, 2]                                                                                                                         
  2                -1  1     18816  yolo.model.impl.common.C3               [64, 64, 1]                                                                                                                            
  3                -1  1     73984  yolo.model.impl.common.Conv             [64, 128, 3, 2]                                                                                                                        
  4                -1  2    115712  yolo.model.impl.common.C3               [128, 128, 2]                                                                                                                          
  5                -1  1    295424  yolo.model.impl.common.Conv             [128, 256, 3, 2]                                                                                                                       
  6                -1  3    625152  yolo.model.impl.common.C3               [256, 256, 3]                                                                                                                          
  7                -1  1   1180672  yolo.model.impl.common.Conv             [256, 512, 3, 2]                                                                                                                       
  8                -1  1   1182720  yolo.model.impl.common.C3               [512, 512, 1]                                                                                                                          
  9                -1  1    656896  yolo.model.impl.common.SPPF             [512, 512, 5]                                                                                                                          
 10                -1  1    131584  yolo.model.impl.common.Conv             [512, 256, 1, 1]                                                                                                                       
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']                                                                                                                   
 12           [-1, 6]  1         0  yolo.model.impl.common.Concat           [1]                                                                                                                                    
 13                -1  1    361984  yolo.model.impl.common.C3               [512, 256, 1, False]                                                                                                                   
 14                -1  1     33024  yolo.model.impl.common.Conv             [256, 128, 1, 1]                                                                                                                       
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']           
 16           [-1, 4]  1         0  yolo.model.impl.common.Concat           [1]                            
 17                -1  1     90880  yolo.model.impl.common.C3               [256, 128, 1, False]           
 18                -1  1    147712  yolo.model.impl.common.Conv             [128, 128, 3, 2]               
 19          [-1, 14]  1         0  yolo.model.impl.common.Concat           [1]                            
 20                -1  1    296448  yolo.model.impl.common.C3               [256, 256, 1, False]           
 21                -1  1    590336  yolo.model.impl.common.Conv             [256, 256, 3, 2]               
 22          [-1, 10]  1         0  yolo.model.impl.common.Concat           [1]                            
 23                -1  1   1182720  yolo.model.impl.common.C3               [512, 512, 1, False]           
 24      [17, 20, 23]  1     67425  yolo.model.impl.detect.Detect           [20, [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], [128, 256, 512]]
YOLOv5s summary: 214 layers, 7073569 parameters, 7073569 gradients, 16.1 GFLOPs

AMP: checks passed ✅
optimizer: SGD(lr=0.01115) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.00031), 60 bias
train: Scanning /home/zj/pp/datasets/VOC/labels/train2007.cache... 16551 images, 0 backgrounds, 0 corrupt: 100%|██████████| 16551/16551 00:00
val: Scanning /home/zj/pp/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 00:00

AutoAnchor: 0.00 anchors/target, 0.001 Best Possible Recall (BPR). Anchors are a poor fit to dataset ⚠️, attempting to improve...
AutoAnchor: Running kmeans for 9 anchors on 40058 points...
AutoAnchor: Evolving anchors with Genetic Algorithm: fitness = 0.7470: 100%|██████████| 1000/1000 00:09
AutoAnchor: thr=0.26: 0.9984 best possible recall, 5.74 anchors past thr
AutoAnchor: n=9, img_size=640, metric_all=0.376/0.747-mean/best, past_thr=0.500-mean: 37,53, 74,78, 87,162, 182,136, 148,292, 251,264, 453,203, 305,449, 527,396
AutoAnchor: Done ✅ (optional: update model *.yaml to use these anchors in the future)
Plotting labels to runs/train/exp25/labels.jpg... 
Image sizes 640 train, 640 val
Using 16 dataloader workers
Logging results to runs/train/exp25
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
...
...
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      99/99      1.21G     0.0158    0.04533   0.008878          8        640: 100%|██████████| 1035/1035 01:29
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 619/619 00:37
                   all       4952      12032      0.782      0.687       0.77      0.518

100 epochs completed in 3.503 hours.
Optimizer stripped from runs/train/exp25/weights/last.pt, 14.4MB
Optimizer stripped from runs/train/exp25/weights/best.pt, 14.4MB

Validating runs/train/exp25/weights/best.pt...
Fusing layers... 
YOLOv5s summary: 157 layers, 7064065 parameters, 0 gradients, 15.9 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 619/619 00:41
                   all       4952      12032      0.782      0.687       0.77      0.518
             aeroplane       4952        285      0.898      0.708      0.819       0.54
               bicycle       4952        337      0.919      0.777       0.88      0.615
                  bird       4952        459      0.818      0.645      0.745      0.461
                  boat       4952        263      0.724      0.631      0.702      0.409
                bottle       4952        469      0.766      0.578      0.638      0.382
                   bus       4952        213      0.807      0.798      0.857      0.697
                   car       4952       1201      0.848      0.827      0.888      0.664
                   cat       4952        358       0.86      0.712      0.837      0.599
                 chair       4952        756      0.662       0.52      0.604      0.363
                   cow       4952        244       0.78      0.566      0.742      0.501
           diningtable       4952        206      0.679      0.707      0.739      0.477
                   dog       4952        489      0.826      0.626      0.787      0.532
                 horse       4952        348      0.838      0.793      0.863      0.593
             motorbike       4952        325       0.86      0.739      0.847      0.566
                person       4952       4528      0.869      0.755       0.86      0.542
           pottedplant       4952        480      0.679      0.404      0.501      0.239
                 sheep       4952        242      0.636      0.729      0.739      0.516
                  sofa       4952        239      0.638      0.711      0.721      0.532
                 train       4952        282      0.815      0.798      0.853      0.581
             tvmonitor       4952        308      0.711      0.721       0.78      0.549
Results saved to runs/train/exp25
```

hyp_evolve.yaml:

```text
# YOLOv5 Hyperparameter Evolution Results
# Best generation: 266
# Last generation: 299
#    metrics/precision,       metrics/recall,      metrics/mAP_0.5, metrics/mAP_0.5:0.95,         val/box_loss,         val/obj_loss,         val/cls_loss
#              0.74479,               0.6643,              0.73447,               0.4791,             0.018449,             0.025529,            0.0054665

lr0: 0.01115
lrf: 0.08284
momentum: 0.98
weight_decay: 0.00031
warmup_epochs: 3.1106
warmup_momentum: 0.87978
warmup_bias_lr: 0.07814
box: 0.0319
cls: 0.39042
cls_pw: 0.56542
obj: 0.96735
obj_pw: 1.1099
iou_t: 0.2
anchor_t: 3.8327
fl_gamma: 0.0
hsv_h: 0.02015
hsv_s: 0.7496
hsv_v: 0.6246
degrees: 0.0
translate: 0.10601
scale: 0.51105
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 0.84024
mixup: 0.07779
copy_paste: 0.0
anchors: 3.1689
```