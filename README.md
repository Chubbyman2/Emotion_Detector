# Emotion_Detector
A facial expression classifier created using OpenCV and large CNN, with numerous techniques used to increase the accuracy of the model.

The dataset was initially retrieved from Kaggle, with a direct link to the files also being provided:

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

https://drive.google.com/drive/folders/1bCFKTkmItIZl73YNUq5QXppLpIdmTgUv

Methods used to Increase Accuracy of the Model:

-Data Generator, used on training set: rotations, shearing, horizontal and vertical shifts, zooms, horizontal flip, shuffling dataset

-He normal initialization for ReLU layers and Xavier (glorot normal) initialization for softmax layer

-Batch Normalization and image padding

-Adam optimizer (lr=0.001)

-ModelCheckpoint, Learning Rate Reduction, Early Stopping


Result:

First Round of Training: Final Accuracy = 47.29%, Validation Accuracy = 38.38%

Second Round of Training: Final Accuracy = 58.33%, Validation Accuracy = 54.99%
