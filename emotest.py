# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:05:55 2022

@author: Toshiba
"""

from keras.models import load_model
from time import sleep 
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import time
import  numpy as np
#import pyttsx3

face_classifier=cv2.CascadeClassifier(r'E:/Toshiba/Emotion_Detection_CNN/haarcascade_frontalface_default.xml')
classifier=load_model(r'E:/Toshiba/Emotion_Detection_CNN/model.h5')
emotion_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprize']

cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    labels=[]
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray, (48,48),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            
            prediction=classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position=(x,y-10)
            #engine=pyttsx3.init('sapi5')
            #voices=engine.getProperty('voices')
            #engine.setProperty('voice', voices[1].id)
            #def speak(audio):
            #    engine.say(audio)
               # engine.runAndWait()
          #  def emotion1():
             #   if label=="Happy":
             #       speak(f"Are u in happy face listen some songs ")
               #     time.sleep(0)
               # if label=="Sad":
               #     speak(f"why are u in sad face go and watch movie")
                #    time.sleep(0)
                #if label=="Neutral":
                   #speak(f"listen some Action  songs sir")
                   #time.sleep(0)
    

            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No faces',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
            
    cv2.imshow("Emotion",frame)
    #emotion1()
    #print(label)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            