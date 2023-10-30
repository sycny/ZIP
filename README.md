# Black-box Backdoor Defense via Zero-shot Image Purification (ZIP)
This repository hosts the code for our NeurIPS'2023 paper '[Black-box Backdoor Defense via Zero-shot Image Purification.](https://arxiv.org/abs/2303.12175)' by Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, Ninghao Liu.

![ZIP](https://github.com/sycny/sycny.github.io/blob/main/images/ZIP.png)
The proposed ZIP backdoor defense framework. In Stage 1, we use a linear transformation, such as blurring, to destroy the trigger pattern in the poisoned image. In Stage 2, we design a guided diffusion process to generate the purified image with a pre-trained diffusion model. Finally, in the purified image, the semantic information from x is kept while the trigger pattern is destroyed.

## Datasets & Pre-trained Model
### Datasets

Except for the CIFAR10, the other datasets require you to download manually and put them in the './datasets/'.

1. [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [GTSRB](https://benchmark.ini.rub.de/)
3. [Imagenette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) 

### Diffusion Model
We use a pre-trained diffusion model provided by OpenAI [Download](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)/[ref](https://github.com/openai/guided-diffusion). Please download it and put it into the './exp/logs/imagenet/'
```
exp/
|__ logs/
    |__ imagenet/
       |__256x256_diffusion_uncond.pt
```
You can change this model to other diffusion models, like [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32).
## A Quick Demo
### Prepared the attacked models and pre-purified images
We provide the attacked ResNet-34 model and the purified test datasets (on Imagenette). You can use the following command to view ZIP defense performance.

Step 1. You need to download the [purified image]() first and create the following path:'./pur/Mode3/Imagenette2/'. Please put the purified images into this path like this:
```
pur/
|__ Mode3/
    |__ Imagenette2/
        |__BadNet/
           |__ Demo/
              |__1.0/
                 |_1/
                    |_train
                    |_...
                    |_val_pois
                    |_...
        |__Blended/
           |__ Demo/
              |__1.0/
                 |_1/
                    |_train
                    |_...
                    |_val
                    |_val_pois
                    |_...
        |__PhysicalBA
           |__ Demo/
              |__1.0/
                 |_1/
                    |_train
                    |_...
                    |_val_pois
                    |_...
        |WaNet
           |__ Demo/
              |__1.0/
                 |_1/
                    |_train
                    |_...
                    |_val_pois
                    |_...        
```
Step 2. You need to download the attacked classification model from [google drive](). And create path:'./Imagenette2_pretrain/'. Then, put the model weight in this fold as follows:
```
Imagenette2_pretrain/
    |__BadNet/
        |__ Res_34_256/
            |__ckpt.pth
    |_Blended/
        |__ Res_34_256/
            |__ckpt.pth
    |_ ...
```

### Test the attack performance

You can use the mode4 in our code to test four kinds of attack performance.
```
python main.py --dataset Imagenette2 --attack_method BadNet --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
python main.py --dataset Imagenette2 --attack_method Blended  --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
python main.py --dataset Imagenette2 --attack_method PhysicalBA  --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
python main.py --dataset Imagenette2 --attack_method WaNet  --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
```

### Test the defense performance

You can use the mode3 in our code to test the defense performance with four attacks.
```
python main.py --dataset Imagenette2 --attack_method BadNet -upctes -upptes --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
python main.py --dataset Imagenette2 --attack_method Blended -upctes -upptes --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
python main.py --dataset Imagenette2 --attack_method PhysicalBA -upctes -upptes --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
python main.py --dataset Imagenette2 --attack_method WaNet -upctes -upptes --img_size 256 --deg Demo --at_threshold 1  --deg_scale 1.0 --img_size 256
```
## Defense with your own attack/model
### Train your own model
```
python main.py --dataset Imagenette2 --attack_method BadNet --testwpoisoned True --testwpurified False --img_size 256
```
### Purified the images and test
```
python main.py --dataset Imagenette2 --attack_method BadNet -pctes -pptes -upctes -upptes --at_threshold 0.98  --deg two_aug --deg_scale 4 --img_size 256
```

## Experiments details

For CIFAR10, we apply both blur and gray-scale conversion as linear transformations. In our setting, we put 64 pieces of 32\*32 image together as one 256\*256 image and then purify. When testing, we will split the big image back to the original size. 

For GTSRB, we only apply blur as the linear transformation. In our setting, we put 64 pieces of 32\*32 image together as one 256\*256 image and then purify. When testing, we will split the big image back to the original size. 

For Imagenette, we only apply blur as the linear transformation. 

## References
If you find this repository useful for your research, please cite the following work.

```
@inproceedings{
shi2023black,
title={Black-box Backdoor Defense via Zero-shot Image Purification},
author={Shi, Yucheng and Du, Mengnan and Wu, Xuansheng and Guan, Zihan and Sun, Jin and Liu, Ninghao},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
}
```

## Acknowledgements
The implementation  of the backdoor attack is based on / inspired by [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox).
The implementation of the diffusion model is based on / inspired by [DDNM](https://github.com/wyhuai/DDNM).




