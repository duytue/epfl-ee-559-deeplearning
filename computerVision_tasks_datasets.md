# Computer Vision

## Tasks
* Classification
* Object detection (class and location)
* Semantic or instance segmentation (labeling exact pixels of object)
* Other (video tracking, pose estimation, 3d reconstruction, denoising, ...)

## Datasets
* MNIST and Fashion-MNIST
* CIFAR10 and CIFAR100
* PASCAL VOC 2012
* ImageNet
* Cityscapes

---
## Image Classification
* Standard convnets (LeNet ??)
* Fully convolutional neural networks (AlexNet)
* Network in network (inception - GoogLeNet)
* Residual networks (ResNet)
## Object Detection
* Classify local regions, at multiple scales and locations (Sliding windows).
* Sermanet et al. (2013): add a regression part to predict object's bounding box.
* Region approaches (evolved from AlexNet): suffer from the cost of the region proposal computation.
* Improved by Ren et al. (2015) in "Faster R-CNN".
* Most famous: YOLO (You Only Look Once) algorithm (Redmon et al. 2015).
## Semantic Segmentation
* Old appoarch: measure similarity between pixels, and group similar pixels togethers -> account poorly for semantic content.
* DL appoarch: pixel classification and reuse networks trained for IClassification by making them convolutional.

---
## Data Loader
* Since some datasets are enormous, we cannot fit them into memory. Samples have to be constantly loaded during training.
### PyTorch DataLoader
> torch.utils.data.DataLoader   

