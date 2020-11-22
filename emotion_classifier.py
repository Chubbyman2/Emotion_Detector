import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

classifier = load_model(
    "C:/Users/charl/OneDrive/Desktop/Machine Learning Stuff/emotion_classifier.h5")

classes = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
