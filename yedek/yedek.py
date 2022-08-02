import numpy as np
import cv2
import os.path
import glob
from tkinter import *
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as Dt

files_masked = glob.glob("Face Detection/Masked/*")
files_unmasked = glob.glob("Face Detection/Unmasked/*")
for f in files_masked:
    os.remove(f)
for f in files_unmasked:
    os.remove(f)

win = Tk()

win.geometry("1150x800")
label = Label(win)
label.grid(row=0, column=0)

count_masked = 0
count_unmasked = 0

with_mask = np.load('dataset/with_mask.npy')
without_mask = np.load('dataset/without_mask.npy')

with_mask = with_mask.reshape(500, 50*50*3)
without_mask = without_mask.reshape(500, 50*50*3)

X = np.r_[with_mask, without_mask]

labels = np.zeros(X.shape[0])
labels[500:] = 1.0
names = {0: 'Maskeli', 1: 'Maskesiz'}

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

dt_model = Dt(max_depth=5).fit(x_train, y_train)

y_pred = dt_model.predict(x_test)

ac = accuracy_score(y_test, y_pred)

haar_data = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX

def createLabel(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image

def put_mask():
    global panelA1, panelA2, panelA3
    global count_masked
    global temp_mask, temp_old_mask
    path1 = 'Face Detection/Masked/face' + str(count_masked-1) + '.jpg'
    path2 = 'Face Detection/Masked/face' + str(count_masked-2) + '.jpg'
    path3 = 'Face Detection/Masked/face' + str(count_masked-3) + '.jpg'
    if len(path1) != 0:
        path1_image = createLabel(path1)
        panelA1 = Label(image=path1_image, height=250, width=250)
        panelA1.image = path1_image
        panelA1.grid(row=0, column=0)
        if count_masked > 1:
            path2_image = createLabel(path2)
            panelA2 = Label(image=path2_image, height=250, width=250)
            panelA2.image = path2_image
            panelA2.grid(row=1, column=0)
            if count_masked > 2:
                path3_image = createLabel(path3)
                panelA3 = Label(image=path3_image, height=250, width=250)
                panelA3.image = path3_image
                panelA3.grid(row=2, column=0)

def put_unmask():
    global panelB1, panelB2, panelB3
    global count_unmasked
    global temp, temp_old
    path1 = 'Face Detection/Unmasked/face' + str(count_unmasked-1) + '.jpg'
    path2 = 'Face Detection/Unmasked/face' + str(count_unmasked-2) + '.jpg'
    path3 = 'Face Detection/Unmasked/face' + str(count_unmasked-3) + '.jpg'
    if len(path1) != 0:
        path1_image = createLabel(path1)
        panelB1 = Label(image=path1_image, height=250, width=250)
        panelB1.image = path1_image
        panelB1.grid(row=0, column=2)
        if count_unmasked > 1:
            path2_image = createLabel(path2)
            panelB2 = Label(image=path2_image, height=250, width=250)
            panelB2.image = path2_image
            panelB2.grid(row=1, column=2)
            if count_unmasked > 2:
                path3_image = createLabel(path3)
                panelB3 = Label(image=path3_image, height=250, width=250)
                panelB3.image = path3_image
                panelB3.grid(row=2, column=2)

def show_frames():
    global count_unmasked
    global count_masked
    global faceArea
    flag, frame = cam.read()
    if flag:
        faces = haar_data.detectMultiScale(frame, 1.05, 20)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            faceArea=face
            pred = dt_model.predict(face)[0]
            n = names[int(pred)]
            if n == 'Maskeli':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                a = x + (w-10)
                b = y + (h/2)
                cv2.rectangle(frame, (x+10, y+h), (int(a), int(b)), (0, 255, 0), 4)
                cv2.imwrite("Face Detection/Masked/face" + str(count_masked) + ".jpg", frame)
                count_masked += 1
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.imwrite("Face Detection/Unmasked/face" + str(count_unmasked) + ".jpg", frame)
                count_unmasked += 1
            cv2.putText(frame, n, (x, y), font, 1, (230, 230, 255), 2)
        if count_masked > 100:
            count_masked = 0
        if count_unmasked > 100:
            count_unmasked = 0
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.grid(row=0, column=1, rowspan=3)
        label.configure(image=imgtk)
        if count_unmasked != 0:
            put_unmask()
        if count_masked != 0:
            put_mask()
        label.after(1, show_frames)

show_frames()
win.mainloop()
