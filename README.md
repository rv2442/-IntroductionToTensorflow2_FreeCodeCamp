# IntroductionToTensorflow2_FreeCodeCamp

__This is used for creating a model and predicting traffic signs, based on the GTSRB (German Traffic Sign Recognition Benchmark) Dataset.__   


## Context
The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. We cordially invite researchers from relevant fields to participate: The competition is designed to allow for participation without special domain knowledge. Our benchmark has the following properties:  

1. Single-image, multi-class classification problem
2. More than 40 classes
3. More than 50,000 images in total
4. Large, lifelike database
 
 
 ## Traning accuracy
![Sign_prediction](https://user-images.githubusercontent.com/69571769/193198324-38a5ed3b-4aa5-477b-b1c2-855e06c3f9c2.jpeg)
 
 ## Validation and Test accuracy Evaluation  
![Validation and Test evaluation](https://user-images.githubusercontent.com/69571769/193198314-16bd8836-bf02-480e-b5f8-87a0d2e99eac.jpeg)

## Single Image Prediction
![Training](https://user-images.githubusercontent.com/69571769/193198328-53894669-94e9-43f6-ba2f-f81a1e008fec.jpeg)


### User Guide

#### Installing Dependencies
```
pip -r install requirements.txt
```  
Once done create a folder inside named training_data and inside that create 2 folders train and val.  
Update the paths in Source Code.    

__Dataset__: [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)  

__Preprocessing__ : First the data was divided with a split_size of 0.1 using ```split_data()``` and test set was ordered using Test.Csv from the dataset with ```order_test_set()```  located in my_utils.py    

### MODEL Architecture 
>CONVOLUTIONAL LAYER  
>MAXPOOL LAYER   
>BATCH NORMALIZATION LAYER
>CONVOLUTIONAL LAYER   
>MAXPOOL LAYER   
>BATCH NORMALIZATION LAYER  
>CONVOLUTIONAL LAYER   
>MAXPOOL LAYER   
>BATCH NORMALIZATION LAYER    
>FLATTEN LAYER
>DENSE LAYER
>DENSE LAYER

### USER GUIDE
__Training and Testing__:  
Run GTSRB.py, with ```TRAIN = True``` and ```TEST = True```

__Predicting Single Image__:  
Run my_predictor.py after updating the image path with the path of the image to be predicted.  

__Note__ in windows folders are numbered in the following sequence,
> 0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34 ...  

If you want to run __mnist_example.py__, just execute the file. The dataset will be downloaded automatically when running for the first time.  




