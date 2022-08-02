import numpy as np  # İmported necessary libraries. / Gerekli kütüphaneleri import ettik.
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
for f in files_masked:  # In here, we deleted our old pictures. / Eski fotoğraflarımızı sildik.
    os.remove(f)
for f in files_unmasked:
    os.remove(f)

win = Tk()  # Creating a new window. / Yeni bir pencere oluşturduk.

win.title('Maske Tespit Uygulaması')  # We do necessary things to our window. / Penceremize gerekli ayarlar yaptık.
win.geometry("1150x800")
label = Label(win)
label.grid(row=0, column=0)

count_masked = 0
count_unmasked = 0

with_mask = np.load('dataset/with_mask.npy')  # Import the pictures from folder. / Fotoğraflarımızı projeye aktardık.
without_mask = np.load('dataset/without_mask.npy')

with_mask = with_mask.reshape(500,
                              50 * 50 * 3)  # Generate a dataset from pictures. / Fotoğraflardan dataset oluşturduk.
without_mask = without_mask.reshape(500, 50 * 50 * 3)

dataset = np.r_[with_mask, without_mask]  # We combined datasets. / Dataset'leri birleştirdik.

labels = np.zeros(dataset.shape[0])
labels[500:] = 1.0
names = {0: 'Maskeli', 1: 'Maskesiz'}  # if statement.

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.25)  # training dataset.

dt_model = Dt(max_depth=5).fit(x_train, y_train)  # we fit x and y's train to get better result
# x ve y trainlerini daha iyi bir sonuç için birleştirdik.

y_pred = dt_model.predict(x_test)

ac = accuracy_score(y_test, y_pred)  # Accuracy score for the similarity. / benzerlik için başarı skoru

haar_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)  # Starting the video record. / Video kaydını başlattık.
data = []
font = cv2.FONT_HERSHEY_COMPLEX


def createLabel(path):  # in this function, we create image for our labels. / Burada label'lar için fotoğraf oluşturduk.
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image


def put_mask():  # in this function, we create 3 label for masked pictures. / Burada maskeli fotoğraflar için 3 label
    global panelA1, panelA2, panelA3  # oluşturduk.
    global count_masked
    global temp_mask, temp_old_mask
    path1 = 'Face Detection/Masked/face' + str(count_masked - 1) + '.jpg'
    path2 = 'Face Detection/Masked/face' + str(count_masked - 2) + '.jpg'
    path3 = 'Face Detection/Masked/face' + str(count_masked - 3) + '.jpg'
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


def put_unmask():  # in this function we create 3 label for unmasked pictures. / Burada maskesiz fotoğraflar için 3
    global panelB1, panelB2, panelB3  # label oluşturduk.
    global count_unmasked
    global temp, temp_old
    path1 = 'Face Detection/Unmasked/face' + str(count_unmasked - 1) + '.jpg'
    path2 = 'Face Detection/Unmasked/face' + str(count_unmasked - 2) + '.jpg'
    path3 = 'Face Detection/Unmasked/face' + str(count_unmasked - 3) + '.jpg'
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


def show_frames():  # in this function we get the real time video capture in main label. / Burada canlı video kaydını
    global count_unmasked  # ana labele aktardık.
    global count_masked
    flag, frame = cam.read()  # starting the read frame by frame. / Videoyu kare kare okumaya başladık.
    if flag:
        faces = haar_data.detectMultiScale(frame, 1.05, 20)  # Face recognition. / Yüz tanıma.
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w, :]  # Picking up the face from the picture. / Fotoğraftan yüzü aldık.
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            pred = dt_model.predict(face)[0]  # Test the face to recognize masked or unmasked. / Maskeli mi değil mi
            n = names[int(pred)]  # diye yüzü test ettik.
            if n == 'Maskeli':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # face in rectangle. / Yüzü kareye aldık
                a = x + (w - 10)
                b = y + (h / 2)
                roi_masked = frame[y:y + h, x:x + w]  # mask in rectangle. / maskeyi kareye aldık.
                cv2.rectangle(frame, (x + 10, y + h), (int(a), int(b)), (0, 255, 0), 4)
                cv2.imwrite("Face Detection/Masked/face" + str(count_masked) + ".jpg", roi_masked)  # Saving. / Kayıt.
                count_masked += 1
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                roi_unmasked = frame[y:y + h, x:x + w]  # Face in rectangle for unmasked face./ Maskesiz yüzü kareye.
                cv2.imwrite("Face Detection/Unmasked/face" + str(count_unmasked) + ".jpg", roi_unmasked)  # Saving
                count_unmasked += 1
            cv2.putText(frame, n, (x, y), font, 1, (230, 230, 255), 2)
        if count_masked > 100:
            count_masked = 0  # for above 100 face, we start from 0 again. / 100 yüzden sonra sıfırdan başlıyor.
        if count_unmasked > 100:  # for avoiding memory issues. / Depoloma sorununu önlemek için.
            count_unmasked = 0
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # because of different color format. / farklı renk formatı.
        img = Image.fromarray(frame)  # in next 5 statement we create label for real time capture. / Önümüzdeki 5 satır
        imgtk = ImageTk.PhotoImage(image=img)  # boyunca canlı video için label oluşturuyoruz.
        label.imgtk = imgtk
        label.grid(row=0, column=1, rowspan=3)
        label.configure(image=imgtk)
        if count_unmasked != 0:
            put_unmask()  # If unmasked photo came, we put this photo to our window with this function.
        if count_masked != 0:
            put_mask()  # Eğer maskeli fotoğraf gelirse, o fotoğrafı bu fonksiyon ile penceremize ekliyoruz.
        label.after(1, show_frames)


show_frames()  # put the window contents in the window. / Pencere içeriklerini penceremize ekle.
win.mainloop()  # until we close, the window stay opened. / Biz kapatana kadar pencere açık kalacak.
