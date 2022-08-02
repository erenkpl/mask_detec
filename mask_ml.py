import cv2
import numpy as np

haar_data = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# we collect haar cascade data from the .xml folder. / haar cascade datalarını .xml dosyasından topladık

cam = cv2.VideoCapture(0)  # Starting the record. / Kayda başladık.
data = []
count = 0

while True:
    flag, frame = cam.read()  # Read frame by frame. / Kare kare okuduk.
    if flag:
        faces = haar_data.detectMultiScale(frame, 1.05, 10)  # Recognize the faces./ Yüzleri tanıdık.
        for x, y, w, h in faces:  # coordinates of faces./ yüzlerin koordinatları.
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 4)  # face in rectangle./ Yüzleri kare içine aldık.
            face = frame[y:y+h, x:x+h, :]  # take face from photo./ Yüzü fotoğraftan aldık.
            face = cv2.resize(face, (50, 50))
            print(len(data))
            if len(data) < 500:  # for 500 faces./ 500 adet yüz için.
                data.append(face)  # We created a data from collected faces./ Aldığımız yüzlerden data oluşturduk.
        cv2.imshow('result', frame)

    np.save('dataset/without_mask.npy', data)  # Saved data to directory./ Belirli bi konuma datayı kaydettik.

    if cv2.waitKey(2) == 27 or len(data) >= 500:  # for exit with "esc" key on the keyboard or exiting with 500. photo.
        break                                     # klavyedeki "esc" tuşuyla ya da 500. fotoğrafla çıkmak için.

cam.release()
cv2.destroyAllWindows()
