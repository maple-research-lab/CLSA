# CLSA

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/CLSA-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>  

CLSA is a self-supervised learning methods which focused on the pattern learning from strong augmentations. 

Copyright (C) 2020 Xiao Wang, Guo-Jun Qi

License: MIT for academic use.

Contact: Guo-Jun Qi (guojunq@gmail.com)


## Introduction
Representation learning has been greatly improved with the advance of contrastive learning methods. Those methods have greatly benefited from various data augmentations that are carefully designated to maintain their identities so that the images transformed from the same instance can still be retrieved. However, those carefully designed transformations limited us to further explore the novel patterns carried by other transformations. To pave this gap, we propose a general framework called Contrastive Learning with Stronger Augmentations(CLSA) to complement current contrastive learning approaches. As found in our experiments, the distortions induced from the stronger make the transformed images can not be viewed as the same instance any more. Thus, we propose to minimize the distribution divergence between the weakly and strongly augmented images over the representation bank to supervise the retrieval of strongly augmented queries from a pool of candidates. Experiments on ImageNet dataset and downstream datasets showed the information from the strongly augmented images can greatly boost the performance. For example, CLSA achieves top-1 accuracy of 76.2% on ImageNet with a standard ResNet-50 architecture with a single-layer classifier fine-tuned, which is almost the same level as 76.5% of supervised results. 

## Installation  
CUDA version should be 10.1 or higher. 
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:maple-research-lab/CLSA.git && cd CLSA
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip install torch==1.7.1
pip install torchvision==0.8.2
pip install numpy==1.19.5
pip install Pillow==5.1.0
pip install tensorboard==1.14.0
pip install tensorboardX==1.7
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n CLSA python=3.6.9
conda activate CLSA
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate CLSA
conda deactivate(If you want to exit) 
```
#### 4 Prepare the ImageNet dataset
##### 4.1 Download the [ImageNet2012 Dataset](http://image-net.org/challenges/LSVRC/2012/) under "./datasets/imagenet2012".
##### 4.2 Go to path "./datasets/imagenet2012/val"
##### 4.3 move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Usage

### Unsupervised Training
This implementation only supports multi-gpu, DistributedDataParallel training, which is faster and simpler; single-gpu or DataParallel training is not supported.
#### Single Crop
##### 1 Without symmetrical loss
```
python3 main_clsa.py --data=[data_path] --workers=32 --epochs=200 --start_epoch=0 --batch_size=256 --lr=0.03 --weight_decay=1e-4 --print_freq=100 --world_size=1 --rank=0 --dist_url=tcp://localhost:10001 --moco_dim=128 --moco_k=65536 --moco_m=0.999 --moco_t=0.2 --alpha=1 --aug_times=5 --nmb_crops 1 1 --size_crops 224 96 --min_scale_crops 0.2 0.086 --max_scale_crops 1.0 0.429 --pick_strong 1 --pick_weak 0 --clsa_t 0.2 --sym 0
```
Here the [data_path] should be the root directory of imagenet dataset.

##### 2 With symmetrical loss (Not verified)
```
python3 main_clsa.py --data=[data_path] --workers=32 --epochs=200 --start_epoch=0 --batch_size=256 --lr=0.03 --weight_decay=1e-4 --print_freq=100 --world_size=1 --rank=0 --dist_url=tcp://localhost:10001 --moco_dim=128 --moco_k=65536 --moco_m=0.999 --moco_t=0.2 --alpha=1 --aug_times=5 --nmb_crops 1 1 --size_crops 224 96 --min_scale_crops 0.2 0.086 --max_scale_crops 1.0 0.429 --pick_strong 1 --pick_weak 0 --clsa_t 0.2 --sym 1
```
Here the [data_path] should be the root directory of imagenet dataset.

#### Multi Crop

##### 1  Without symmetrical loss
```
python3 main_clsa.py --data=[data_path] --workers=32 --epochs=200 --start_epoch=0 --batch_size=256 --lr=0.03 --weight_decay=1e-4 --print_freq=100 --world_size=1 --rank=0 --dist_url=tcp://localhost:10001 --moco_dim=128 --moco_k=65536 --moco_m=0.999 --moco_t=0.2 --alpha=1 --aug_times=5 --nmb_crops 1 1 1 1 1 --size_crops 224 192 160 128 96 --min_scale_crops 0.2 0.172 0.143 0.114 0.086 --max_scale_crops 1.0 0.86 0.715 0.571 0.429 --pick_strong 0 1 2 3 4 --pick_weak 0 1 2 3 4 --clsa_t 0.2 --sym 0
```
Here the [data_path] should be the root directory of imagenet dataset.

##### 2 With symmetrical loss (Not verified)
```
python3 main_clsa.py --data=[data_path] --workers=32 --epochs=200 --start_epoch=0 --batch_size=256 --lr=0.03 --weight_decay=1e-4 --print_freq=100 --world_size=1 --rank=0 --dist_url=tcp://localhost:10001 --moco_dim=128 --moco_k=65536 --moco_m=0.999 --moco_t=0.2 --alpha=1 --aug_times=5 --nmb_crops 1 1 1 1 1 --size_crops 224 192 160 128 96 --min_scale_crops 0.2 0.172 0.143 0.114 0.086 --max_scale_crops 1.0 0.86 0.715 0.571 0.429 --pick_strong 0 1 2 3 4 --pick_weak 0 1 2 3 4 --clsa_t 0.2 --sym 1
```
Here the [data_path] should be the root directory of imagenet dataset.

### Linear Classification
With a pre-trained model, we can easily evaluate its performance on ImageNet with:
```
python3 lincls.py --data=./datasets/imagenet2012 --dist-url=tcp://localhost:10001 --pretrained=[pretrained_model_path]
```
[pretrained_model_path] should be the Imagenet pretrained model path.

Performance:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>network</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">Crop</th>
<th valign="bottom">CLSA<br/>top-1 acc.</th>
<th valign="bottom">Model<br/>Link</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Single</td>
<td align="center">69.4</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EVm8m4TUhDJBtouNUDv-x4UBENdkHvRMNfU12Mm9G_HkIQ?e=8dk0wN">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Multi</td>
<td align="center">73.3</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/Ed8IVMBAvp1GmqABFMskEbYBz6B1vq65kp2IQlukFiS6mw?e=K0G5H6">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">72.2</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/Ear2V9zCe9dEvT0mr_J-BgoBsRBDRau0n9oGVDtBBzFl8w?e=o2ZWEM">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Multi</td>
<td align="center">76.2</td>
<td align="center">None</td>
</tr>
</tbody></table>

Really sorry that we can't provide CLSA* 800 epochs' model, which is because that we train it with 32 internal GPUs and we can't download it because of company regulations. For downstream tasks, we found multi-200epoch model also had similar performance. Thus, we suggested you to use this [model](https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/Ed8IVMBAvp1GmqABFMskEbYBz6B1vq65kp2IQlukFiS6mw?e=K0G5H6) for downstream purposes.

### Transfering to VOC07 Classification
#### 1 Download [Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) under "./datasets/voc"
#### 2 Linear Evaluation:
```
cd VOC_CLF
python3 main.py --data=[VOC_dataset_dir] --pretrained=[pretrained_model_path]
```
Here VOC directory should be the directory includes "vockit" directory; [VOC_dataset_dir] is the VOC dataset path; [pretrained_model_path] is the imagenet pretrained model path.

### Transfer to Object Detection
#### 1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

#### 2. Convert a pre-trained CLSA model to detectron2's format:
   ```
   # in detection folder
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

#### 3. download [VOC Dataset](http://places.csail.mit.edu/user/index.php) and [COCO Dataset](https://cocodataset.org/#download) under "./detection/datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets) requried by detectron2.

#### 4. Run training:
##### 4.1 Pascal detection
   ```
   cd detection
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_CLSA.yaml  --num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```
##### 4.2 COCO detection
```
   cd detection
   python train_net.py --config-file configs/coco_R_50_C4_2x_clsa.yaml --num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```


## Citation:
[Contrastive Learning with Stronger Augmentations](https://arxiv.org/abs/2104.07713)
```
@article{wang2021CLSA,
  title={Contrastive Learning with Stronger Augmentations},
  author={Wang, Xiao and Qi, Guo-Jun},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```


