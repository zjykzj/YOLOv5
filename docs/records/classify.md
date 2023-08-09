
# Classify


## YOLOv5s with ImageNet

```text
python -m torch.distributed.run --nproc_per_node 4 --master_port 25123 classify/train.py --model /home/zj/pp/YOLOv5/runs/cls/exp/yolov5s-cls.pt --data imagenet --img 224 --epochs 90 --device 0,1,2,3
...
...
     90/90    0.889G        2.79        2.52       0.649        0.86: 100%|██████████| 20019/20019 35:10

Training complete (38.831 hours)
Results saved to runs/train-cls/exp21
Predict:         python classify/predict.py --weights runs/train-cls/exp21/weights/best.pt --source im.jpg
Validate:        python classify/val.py --weights runs/train-cls/exp21/weights/best.pt --data /home/zj/pp/datasets/imagenet
Export:          python export.py --weights runs/train-cls/exp21/weights/best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train-cls/exp21/weights/best.pt')
Visualize:       https://netron.app
```

```shell
$ python classify/val.py --weights runs/train-cls/exp21/weights/best.pt --data ../datasets/imagenet --img 224                                                                                                                                       
val: data=../datasets/imagenet, weights=['runs/train-cls/exp21/weights/best.pt'], batch_size=128, imgsz=224, device=, workers=8, verbose=True, project=runs/val-cls, name=exp, exist_ok=False, half=False, dnn=False                                                              
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.                                                                                                                                                                                                        
YOLOv5 🚀 2023-8-8 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                  
Fusing layers...                                                                                                                                                                                                                                                                  
Model summary: 86 layers, 5447688 parameters, 0 gradients, 11.4 GFLOPs                                                                                                                                                                                                            
validating: 100%|██████████| 391/391 00:34                                                                                                                                                                                                                                        
                   Class      Images    top1_acc    top5_acc                                                                                                                                                                                                                      
                     all       50000       0.649        0.86  
...
...
Speed: 0.1ms pre-process, 0.2ms inference, 0.0ms post-process per image at shape (1, 3, 224, 224)
Results saved to runs/val-cls/exp7
```

```shell
$ python classify/predict.py --weights runs/train-cls/exp21/weights/best.pt --source assets/imagenet-val/ --imgsz 224
predict: weights=['runs/train-cls/exp21/weights/best.pt'], source=assets/imagenet-val/, data=configs/data/coco128.yaml, imgsz=[224, 224], device=, view_img=False, save_txt=False, nosave=False, augment=False, visualize=False, update=False, project=runs/predict-cls, name=exp, exist_ok=False, half=False, dnn=False, vid_stride=1
requirements: /home/zj/pp/YOLOv5/requirements.txt not found, check failed.
YOLOv5 🚀 2023-8-8 Python-3.8.16 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)

Fusing layers... 
Model summary: 86 layers, 5447688 parameters, 0 gradients, 11.4 GFLOPs
image 1/2 /home/zj/pp/YOLOv5/assets/imagenet-val/ILSVRC2012_val_00016035.JPEG: 224x224 torch 0.84, stage 0.12, electric guitar 0.01, racket 0.00, banjo 0.00, 7.5ms
image 2/2 /home/zj/pp/YOLOv5/assets/imagenet-val/ILSVRC2012_val_00033217.JPEG: 224x224 billiard table 0.51, bell pepper 0.24, maraca 0.04, abacus 0.04, pill bottle 0.02, 6.3ms
Speed: 0.3ms pre-process, 6.9ms inference, 2.4ms NMS per image at shape (1, 3, 224, 224)
Results saved to runs/predict-cls/exp9
```

## 