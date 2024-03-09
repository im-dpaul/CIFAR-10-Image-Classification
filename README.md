# CIFAR-10 Image Classification

This project aims to perform image classification using the CIFAR-10 dataset. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a widely used dataset for machine learning and computer vision tasks, consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)

## Introduction
Image classification is a fundamental task in computer vision, where the goal is to assign a label to an image based on its visual content. In this project, we utilize the CIFAR-10 dataset and deep learning techniques to train a model capable of accurately classifying images into one of ten predefined categories.

## Dataset
The CIFAR-10 dataset consists of 50,000 training images and 10,000 test images, each labeled with one of the following classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is loaded using TensorFlow's `cifar10.load_data()` function, providing access to both the training and test sets.

## Preprocessing
Prior to training, the image data is preprocessed as follows:
- Normalization: Image pixel values are scaled to the range [0, 1].
- One-Hot Encoding: Class labels are converted to categorical format using one-hot encoding using `tensorflow.keras.utils.to_categorical`.

## Model Architecture
The convolutional neural network (CNN) architecture utilized for image classification consists of the following layers:
1. Two sets of Convolutional and MaxPooling layers.
2. Flattening layer to convert 2D feature maps to 1D vectors.
3. Dense hidden layer with 256 neurons and ReLU activation function.
4. Output layer with 10 neurons (equal to the number of classes) and softmax activation for classification.

## Training
The model is compiled using categorical crossentropy loss and the RMSprop optimizer. Early stopping with a patience of 3 epochs is employed to prevent overfitting. The training is performed for 15 epochs with the validation data specified.

## Running the Project
1. Install required libraries: `tensorflow`, `pandas`, `numpy`, `seaborn`, `matplotlib`.
2. Modify hyperparameters (epochs, neurons) in the code if desired.
3. Run the script to train, evaluate, and visualize the model's performance.
