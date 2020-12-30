# VICTR: Visual Information Captured Text Representation for Text-to-Image Multimodal Tasks
This repository contains code for paper [VICTR: Visual Information Captured Text Representation for Text-to-Image Generation Tasks](https://arxiv.org/pdf/2010.03182.pdf)

<h3 align="center">
  <b>Han, C.*, Long, S.*, Luo, S., Wang, K., & Poon, J. (2020, December). <br/><a href="https://www.aclweb.org/anthology/2020.coling-main.277.pdf">VICTR: Visual Information Captured Text Representation for Text-to-Vision Multimodal Tasks</a><br/>In Proceedings of the 28th International Conference on Computational Linguistics (pp. 3107-3117)</b></span>
</h3>


![image](https://github.com/usydnlp/VICTR/blob/main/imgs/arch.jpg)

## 1. Introduction
The proposed VICTR representation for text-to-image multimodal tasks contains two major types of embedding: (1) Basic Graph embedding (for object, relation, attribute) and (2) Positional Graph embedding (for object, relation), which captures rich visual semantic information of objects from the text description. This repository provides the the integration of proposed VICTR representation based on the three original text-to-image generation models: [stackGAN](https://github.com/hanzhanggit/StackGAN), [attnGAN](https://github.com/taoxugit/AttnGAN) and [DM-GAN](https://github.com/MinfengZhu/DM-GAN/tree/master).

## 2. Main code structure and running requirement
Root    ---> repository

>code    ---> the main code for the three models

>>stackgan_victr    ---> main code for stackGAN+VICTR

>>attngan_victr    ---> main code for attnGAN+VICTR

>>dmgan_victr    ---> main code for DM-GAN+VICTR

>DAMSMencoders    ---> pretrained DAMSM text/image encoder from attnGAN

>data

>>coco    ---> COCO2014 images and related data files

>>>train    ---> train related data files

>>>test    ---> test related data files

>output    ---> model output 

Environment for running the code:

- python 3.6

- pytorch 1.4.0 (pip install torch==1.4.0 torchvision==0.5.0)



## 3. Setup and data preperation
### 3.1 Origianl text-to-image related setup
#### ->Preprocessed COCO metadata 
[COCO](https://drive.google.com/file/d/1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9/view) provided by [attnGAN](https://github.com/taoxugit/AttnGAN)

- download and unzip it to ```data/```

#### ->Pretrained DAMSM text/image encoder
  
[DAMSM for COCO](https://drive.google.com/file/d/1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ/view) provided by [attnGAN](https://github.com/taoxugit/AttnGAN)

- download and unzip it to ```DAMSMencoders/```

- for training of the DAMSM model, please refer to [attnGAN](https://github.com/taoxugit/AttnGAN)

### 3.2 Coco2014 images for training and evaluation
Training: ```wget http://images.cocodataset.org/zips/train2014.zip```

Evaluation: ```wget http://images.cocodataset.org/zips/val2014.zip```

- After downloading, unzip all the images under ```data/coco/images/``` folder



### 3.3 Preprocessed caption graphs and trained embeddings of VICTR
Processed caption graphs:
- Training: ``` python google_drive.py 1_Gm_IUzG3U4vWN8Ngv5XPoQgnbQeq5Ld victr_sg_train.zip``` download and unzip to ```data/coco/train/```
- Evaluation: ``` python google_drive.py 1tlNp7nraRmwqoq8LS_7yLRbvweCmdWJL victr_sg_test.zip``` download and unzip to ```data/coco/test/```

Trained graph embeddings: ``` python google_drive.py 1lr7Mcw6R6cr5zYnjYJ_ckmnkR0ARYa3q victr_graph.zip``` download and unzip to ```data/coco/```

## 4. Training

Go to the main code directory of the corresponding model and fun the training command:

- attnGAN-VICTR: ```cd code/attngan_victr``` and ```python main.py --cfg cfg/coco_attn2.yml --gpu 0 --use_sg```

- DM-GAN-VICTR: ```cd code/dmgan_victr``` and ```python main.py --cfg cfg/coco_DMGAN.yml --gpu 0 --use_sg```


The saved models will be available in output files. The training epoch and saving interval can be changed by specifying the value for ```MAX_EPOCH``` and ```TRAIN.SNAPSHOT_INTERVAL``` in the corresponding training yml files:

- attnGAN-VICTR: ```code/attngan_victr/cfg/coco_attn2.yml```

- DM-GAN-VICTR: ```code/dmgan_victr/cfg/coco_DMGAN.yml```

## 5. Evaluation

1. Replace the path to saved models to the ```TRAIN.NET_G``` and ```TRAIN.SG_ATTN``` in the evaluation yml files (e.g. ```NET_G: '../models/netG_epoch_128.pth' ``` and ```SG.SG_ATTN: '../models/attnsg_epoch_128.pth'```), and make sure the ```B_VALIDATION``` is set to ```True``` which will use the coco2014 eval set for generation:

- attnGAN-VICTR: ```code/attngan_victr/cfg/eval_coco.yml```

- DM-GAN-VICTR: ```code/dmgan_victr/cfg/eval_coco.yml```

2. Run the following command:

```python main.py --cfg cfg/eval_coco.yml --gpu 0 --use_sg``` 

3. Evaluation metrics
By running the evaluation code, the generated images can be found in the folder under the model path. To evalute the generated images, the R-precision will be calculated automatically during the evaluation (Using the evaluation code from [DM-GAN](https://github.com/MinfengZhu/DM-GAN)). For the IS and FID, we also use directly the evaluation script from [DM-GAN](https://github.com/MinfengZhu/DM-GAN).

## References:
- [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf) [[github]](https://github.com/hanzhanggit/StackGAN)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) [[github]](https://github.com/taoxugit/AttnGAN)
- [DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis](https://arxiv.org/pdf/1904.01310.pdf) [[github]](https://github.com/MinfengZhu/DM-GAN)
