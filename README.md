# Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing [TIP 2021]

[Hayat Ullah](https://scholar.google.com.pk/citations?user=xnXPj0UAAAAJ&hl=en),
[Khan Muhammad](https://scholar.google.com.pk/citations?user=k5oUZyQAAAAJ&hl=en),
[Muhammad Irfan](https://scholar.google.com.pk/citations?user=MfhBA7EAAAAJ&hl=en),
[Saeed Anwar](https://scholar.google.com.pk/citations?user=vPJIHywAAAAJ&hl=en),
[Muhammad Sajjad](https://scholar.google.com.pk/citations?user=E4-dElIAAAAJ&hl=en),
[Ali Shariq Imran](https://scholar.google.com.pk/citations?user=c75cdz8AAAAJ&hl=en&oi=ao)
[Victor Hugo C. de Albuquerque](https://scholar.google.com.pk/citations?user=meI2k88AAAAJ&hl=en)

[![paper](https://img.shields.io/badge/IEEE|TIP-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/abstract/document/9562276)

<hr />

> **Abstract:**
>*Due to the rapid development of artificial intelligence technology, industrial sectors are revolutionizing in automation, reliability, and robustness, thereby significantly increasing quality and productivity. Most of the surveillance and industrial sectors are monitored by visual sensor networks capturing different surrounding environment images. However, during tempestuous weather conditions, the visual quality of the images is reduced due to contaminated suspended atmospheric particles that affect the overall surveillance systems. To tackle these challenges, this article presents a computationally efficient lightweight convolutional neural network referred to as Light-DehazeNet (LD-Net) for the reconstruction of hazy images. Unlike other learning-based approaches, which separately measure the transmission map and the atmospheric light, our proposed LD-Net jointly estimates both the transmission map and the atmospheric light using a transformed atmospheric scattering model. Furthermore, a color visibility restoration method is proposed to evade the color distortion in the dehaze image. Finally, we conduct extensive experiments using synthetic and natural hazy images. The quantitative and qualitative evaluation on different benchmark hazy datasets verify the superiority of the proposed method over other state-of-the-art image dehazing techniques. Moreover, additional experimentation validates the applicability of the proposed method in the object detection tasks. Considering the lightweight architecture with minimal computational cost, the proposed system is encouraged to be incorporated as an integral part of the vision-based monitoring systems to improve the overall performance.*


![](readme_images/framework.png)

## Introduction
This repo contains the implementation of our proposed LD-Net, the prerequisite libraries, and link of the dataset we have used in our experiments. For testing purpose we also provide a trained LD-Net weights and test hazy images. 

## Installation

This code is written with Anaconda python 3.7. Install the Anaconda python 3.7 and clone the repo with the following command:
```
git clone https://github.com/hayatkhan8660-maker/Light-DehazeNet.git
cd Light-DehazeNet
```

### Dependencies
This code requires the following libraries, install these libraries before testing th code. 
- torch==1.9.1
- torchvision==0.10.1
- numpy==1.19.2
- PIL==8.2.0
- matplotlib==3.3.4

run ```pip install -r requirements.txt``` to install all the dependencies. 

### Training Dataset
For training we have used the Synthetic Hazy Dataset generated by [Boyi Li](https://sites.google.com/site/boyilics/website-builder/project-page) and his team for single image dehazing task. The dataset can be downloaded using the following links. 

- The synthetic hazy images [training set](https://drive.google.com/file/d/17ZWJOpH1AsYQhoqpWR6PK61HrUhArdAK/view)
- The original images [training set](https://drive.google.com/file/d/1Sz5ZFFZXo3sY85R3v7yJa6W6riDGur46/view)

The structure of the dataset directory should be as follows:
```
data
├── original_images
│   ├── images
│       ├── data
│       ├── NYU2_1.jpg
│       ├── NYU2_2.jpg
│       ├── NYU2_3.jpg
│       ├── .....
├── training_images
│   ├── data
│       ├── data
│       ├── NYU2_1_1_2.jpg
│       ├── NYU2_1_1_3.jpg
│       ├── NYU2_1_2_1.jpg
│       ├── .....
   
```

## Training
Run the following command to retrain the LD-Net model on Synthic hazy images from NYU2 Hazy dataset.
```
python run_experiment.py -th data/training_images/data/ -to data/original_images/images/ -e 60 -lr 0.0001
```

## Inferencing 
For testing purpose, we provide two different inferencing scripts namely single_test_inference.py and muliple_test_inference.py. Here single_test_inference.py performs dehazing on single image, whereas muliple_test_inference.py dehaze a batch of images at once as per user's requirements.

### Single_Image_Inferencing 
To run the single_test_inference.py, type the following command in terminal:
```
python single_test_inference.py -i query_hazy_images/outdoor_synthetic/soh(5).jpg
```

### Multiple_Images_Inferencing 
To run the muliple_test_inference.py, type the following command in terminal:
```
python muliple_test_inference.py -td query_hazy_images/outdoor_natural/
```

## Experiments
Originally the LD-Net model is implemented in Python 3.7 and PyTorch 1.9.1 and trained on a PC having NVIDIA GTX 1060 6GB GPU with CUDA 10.0. If you want to retrain the LD-Net model or reproduce the dehazing results make sure the system meets the aforementioned experimental settings. 

### Obtained Visual Results
![](readme_images/insynres.png)
![](readme_images/ousynres.png)
![](readme_images/ounatres.png)
![](readme_images/rgbsdres.png)

### CVR Module :warning:
Oringinally the CVR module is written in matlab. For user's convenience, we are working to parse it to Python and soon will upload it to this repo.   
For time being the CVR matlab version can be accessed with the following google drive link : (https://drive.google.com/file/d/12myYdZbZqegrORpzn62CEo_pqJwbNaU6/view?usp=sharing)



## Citation
Please cite our paper, if you want to reproduce the results using this code.
```
@article{ullah2021light,
  title={Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing},
  author={Ullah, Hayat and Muhammad, Khan and Irfan, Muhammad and Anwar, Saeed and Sajjad, Muhammad and Imran, Ali Shariq and De Albuquerque, Victor Hugo C},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```

```
Ullah, H., Muhammad, K., Irfan, M., Anwar, S., Sajjad, M., Imran, A. S., & De Albuquerque, V. H. C., 
"Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing", 
IEEE Transactions on Image Processing, 2021.
```
