import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = []
for i in os.listdir(r'E:\ISGA\python\OpenCv\faceDetection\pictures\trainingSet'):
    people.append(i)

features = np.load('featurs.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(
    r'E:\ISGA\python\OpenCv\faceDetection\pictures\testingSet\justin bieber\70344761e1e53ef7bd68591dff038bfd.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('before', gray)

# detect the face
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

i = 0
for (x, y, w, h) in faces_rect:
    # face ragion of intrest
    i += 1
    faces_roi = gray[y:y + h, x:x + w]
    cv.imshow('exactFace {}'.format(i), faces_roi)
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'name : {people[label]} with conf : {confidence}')

    cv.putText(img, str(people[label]), (x, y - 5), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('detected face : ', img)
cv.waitKey(0)
