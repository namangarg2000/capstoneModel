import apirequests
import New_main
import cv2
import json
import face_recognition
#from matplotlib import pyplot as plt
#from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from imutils.video import VideoStream
#import imutils#for basic image processing#imutils is Python basic image processing functional package
#from datetime import datetime
#import math
#import os
#import sys
#from threading import Timer
#import shutil
import time
from pathlib import Path
from os.path import dirname, join
print("here1")
#path = cv2.data.haarcascades
#face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')#class to detect objects in a video stream
#eye_cascade = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')


#myList = os.listdir(img_path)#images folders
#processing dataset1


#plt.imshow()
#for img in images:
def   detect_and_predict_mask(frame, faceNet, maskNet,threshold):
	# grab the dimensions of the frame and then construct a blob
	# from it
	global detections 
	(h, w) = frame.shape[:2]
    #Create the four-dimensional blob
    #In order to make the inference from the pre-trained models in OpenCV, the images are first preprocessed using function blobFromImages() or blobFromImage() which will output a blob. This blob is then provided as input to the trained model to get the inference output.
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))#Blob is a 4-D tensor (NCHW) with N: number of image(s), C: number of channel(s), H: height of the image(s) and W: width of the image(s).
	# pass the blob through the network and obtain the face detections
    #Setting blob as input to the network
	faceNet.setInput(blob)
    #Forward pass through the network
	detections = faceNet.forward() #numpy array showing probability value for each class.

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	locs = []
	preds = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >threshold:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
            
			# add the face and bounding boxes to their respective
			# lists
			#locs.append((startX, startY, endX, endY))
			#print(maskNet.predict(face)[0].tolist())
			preds.append(maskNet.predict(face)[0].tolist())
	return (locs, preds)
# SETTINGS
MASK_MODEL_PATH=os.path.join(str(Path(__file__).parent) ,"masksdetection-master\model\Mask_Model.h5")
print("here2")
#FACE_MODEL_PATH="C:\masksdetection-master\masksdetection-master\face_detector"
#SOUND_PATH="C:\masksdetection-master\masksdetection-master\sounds\alarm.wav"
THRESHOLD = 0.7

# Load Sounds
#mixer.init()
#sound = mixer.Sound(SOUND_PATH)


protoPath = join(dirname(__file__), "deploy.prototxt")
weightsPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
print("here3")
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
#prototxtPath = "C:\masksdetection-master\masksdetection-master\face_detector\deploy.prototxt.txt"
#weightsPath = os.path.sep.join([FACE_MODEL_PATH,"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(protoPath, weightsPath)#reads the network model#trained model
print("here4")
# load the face mask detector model from disk
#print("[INFO] loading face mask detector model...")
maskNet = load_model(MASK_MODEL_PATH)
print("here5")
# initialize the video stream and allow the camera sensor to warm up
#print("[INFO] starting video stream...")
#vs = VideoStream(0).start()
#time.sleep(2.0)


print("here6")
def markEnterance(name):
     with open('data.json','w') as f:
                now = time.time()
                dtString = str(now)
                aDict = {"rollno": name, "Time": dtString}
                jsonString = json.dumps(aDict)
                f.write(jsonString)
                f.close()
     apirequests.test(jsonString)
     return True 


#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)#open webcam and capture face#to specify the video source (for example cv2.CAP_DSHOW on Windows).

def recog():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #currentStartTime = time.time()
    #print(currentStartTime)
    #while time.time() - currentStartTime <= 5.0:
    jsonwrite = False
    #counter = 0
    while not jsonwrite:
        success , img = cap.read()
        print(img,jsonwrite)
        imgs = cv2.resize(img,(0,0),None,0.25,0.25)
        imgs = cv2.cvtColor(imgs , cv2.COLOR_BGR2RGB)
        (locs, preds) = detect_and_predict_mask(imgs, faceNet, maskNet,THRESHOLD)
        facesCurFrame = face_recognition.face_locations(imgs)
        encodeCurFrame  = face_recognition.face_encodings(imgs,facesCurFrame)
        
        for encodeFace , faceLoc , pred in zip(encodeCurFrame,facesCurFrame ,preds):
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            #matches = face_recognition.compare_faces(encode_list, encodeFace)
            faceDis = face_recognition.face_distance(New_main.encode_list, encodeFace)
            matchIndex = np.argmin(faceDis)
            name = New_main.class_names[matchIndex]
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4 , x2*4 , y2*4 , x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),color, cv2.FILLED)
            cv2.putText(img, name, (x1+6 , y2 - 6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2 )
            cv2.putText(img, label, (x1 , y2+10) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,0,255) , 2)
           
            jsonwrite = markEnterance(name)
            cv2.imshow('Project' , img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
print("here7")
recog()
print("here8")
cv2.destroyAllWindows()
