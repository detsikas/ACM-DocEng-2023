# ACM-DocEng-2023
Python Tensorflow model for the participation to the ACM DocEng 2023 competition on Document Binarization.

## Prediction
By executing
```
python predict.py <input-image> <output-image> 
```
the binarization model is applied to the *input-image* and the result is saved to the *output-image*

The user can get help on executing the prediction by doing
```
python predict.py --help
```

## Unit testing
In order to verify that the repository has been properly cloned and prediction runs as expected, the user can execute the basic unit test by running
```
python unit_test.py
```

## Project structure
The project contains the Tensorflow model for the ** Dilated Multires Visual Attention Unet** binarization deep learning architecture.
Only the prediction code is included, for the purposes of the competition.

## Model complexity
```
Total params: 6,543,132
Trainable params: 6,519,172
Non-trainable params: 23,960
```

## Environment
The code runs without issues with 
* python 3.8.10
* tensorflow 2.10.0
* opencv-python 4.7.0.72

All the requirements are listed in **requirements.txt**.




