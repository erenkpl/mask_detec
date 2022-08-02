import numpy as np
import cv2
import os.path
from imutils import face_utils
import matplotlib.pyplot as plt
import os, os.path

masked_len = len([name for name in os.listdir('Face Detection/Masked/')])
unmasked_len = len([name for name in os.listdir('Face Detection/Unmasked/')])

count = 0
temp = 0

gray = plt.imread('test_face.jpg')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("temp", temp)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        if w > 150:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            smile = mouthCascade.detectMultiScale(roi_gray, scaleFactor=1.16, minNeighbors=28, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            eyes = eyeCascade.detectMultiScale(roi_gray)

            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
                cv2.putText(frame, 'Mouth', (x + sx, y + sy), 1, 1, (0, 255, 0), 1)

            if len(smile) == 0:
                if str(len(faces)) != temp:
                    cv2.imwrite("Face Detection/Masked/face" + str(count + masked_len) + ".jpg", frame)
                    temp = 0
            if len(smile) == 1:
                if str(len(faces)) != temp:
                    cv2.imwrite("Face Detection/Unmasked/face" + str(count + unmasked_len) + ".jpg", frame)
                    temp = 0

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(frame, 'Eye', (x + ex, y + ey), 1, 1, (0, 255, 0), 2)
        temp = str(len(faces))

    cv2.putText(frame, 'Number of faces : ' + str(len(faces)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
