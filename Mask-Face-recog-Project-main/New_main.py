import apirequests
import cv2
import json
import face_recognition
from matplotlib import pyplot as plt
#from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
#import imutils#for basic image processing#imutils is Python basic image processing functional package
from datetime import datetime
#import math
import os
#import sys
#from threading import Timer
#import shutil
import time
from pathlib import Path
from os.path import dirname, join
img_path = os.path.join(str(Path(__file__).parent) ,"Dataset_")
images = []
class_names = []
encode_list = []
encode_list_cl = []
for subdir in os.listdir(img_path):
#subdir = "Unknown"
    path = img_path + '\\' + subdir #path of images folder in dataset1
    path = path + '\\'
    for img in os.listdir(path):
        img_pic = path + img
        class_names.append(subdir)
        cur_img = cv2.imread(img_pic)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)#to convert an image from one color space to another
        images.append(cur_img)
def find_encodings(images) :
        for img in images :
            encodings = face_recognition.face_encodings(img)[0]#Given an image, return the 128-dimension face encoding for each face in the image
            encode_list.append(encodings)#we can measure the similarity between two face images — that can tell us if they belong to the same person.
            #the information obtained out of the image — that is used to identify the particular face.
        return encode_list
encodeListKnown = find_encodings(images)
