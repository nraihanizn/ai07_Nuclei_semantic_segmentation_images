Nuclei Semantic Segmentation Images

## 1. Overview
Ther purpose of this project is to identify a range of nuclei across varied conditions.  Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work. This project use the dataset from [2018 Data Science Bowl](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and the IDE used to create this project is Spyder. The framewok used is Pandas, Scikit-learn and TensorFlow Keras. 

## 2. Methodology
### Data preprocessing
The images that was downloaded from kaggle contain images that was separated into train and test folder with each has the nuclei image and the mask. The data are resized into (128,128). The image are normalize and process into more suitable form that is prefetch data for the efficiency, splitted int 80:20 train/test set. 

### Model

The model are based on the U_net with modification of a pre-trained MobileNetV2 as the downsampler. The MobileNetV2 will be freeze and not to be trained. A series of block will act as the upsampler as per diagram below. Augmentation layer will be applied as the input and the ouput is a conv transpose layer where it has output size of (128,128)

![u-net-architecture](https://user-images.githubusercontent.com/92585515/182058497-86f93d80-bf4f-49d1-be87-2a98cb0ca91b.png)

The structure of the modified U-net model is

![the modified U-net model](https://user-images.githubusercontent.com/92585515/182060510-30cac5e6-ccfc-4171-8715-7a5264c2046c.png)


### Result
Train for 20 epoch with loss of 0.089 and accuracy of 0.964

![nuclei semantic segmentation result1](https://user-images.githubusercontent.com/92585515/182061521-c0fb42bc-5b0a-40e2-a56f-cfeab4e67835.png)


![nuclei semantic segmentation result2](https://user-images.githubusercontent.com/92585515/182061541-d7a6806a-c8f7-4629-a2c9-e8977f8dace0.pn
