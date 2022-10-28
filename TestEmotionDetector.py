import matplotlib.pyplot as plt
import numpy as np
#import os
import cv2
import pathlib 


from tensorflow.keras.utils import img_to_array #, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
classifier =load_model(r'C:\Users\shrey\Desktop\AI\training\images\trained_model.h5')

camera = cv2.VideoCapture(0)
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

while True:
  _, frame = camera.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40), flags= cv2.CASCADE_SCALE_IMAGE)
  

  for(x, y, width, height) in faces:
     cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
     gray_img = gray[y:y+height, x:x+width]
     gray_img = cv2.resize(gray_img,(48,48),interpolation=cv2.INTER_AREA)
     
     if np.sum([gray_img])!=0:
            roi = gray_img.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
  
            prediction = classifier.predict(roi)[0]  
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y-30)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
     else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    

  cv2.imshow("Faces", frame)
  if cv2.waitKey(1) == ord("q"):
       break

camera.release()
cv2.destroyAllWindows()
