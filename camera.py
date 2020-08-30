import cv2
import numpy as np
from predictor import predict
facec = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        faces = facec.detectMultiScale(fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (100, 100))
            pred = predict(roi)

            
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()