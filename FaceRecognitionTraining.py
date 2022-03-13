import os
import numpy as np
import cv2 as cv

DIR = r'E:\ISGA\python\OpenCv\faceDetection\pictures\trainingSet'

people = []

for i in os.listdir(DIR):
    people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')


featurs = []
labels = []
# loop in every folder and grab faces and
# add them into the training set which has featurs and corisponding lable
def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)
            for (x,y,w,h) in faces_rect:
                # face ragion of intrest
                faces_roi = gray[y:y+h, x:x+w]
                featurs.append(faces_roi)
                labels.append(label)

create_train()
print(f'length of the featurs list is : {len(featurs)}, labels = {len(labels)}')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

print('training ---------------------------------100% finished')
featurs = np.array(featurs, dtype='object')
labels = np.array(labels)
# tarining on featurs and labels

face_recognizer.train(featurs,labels)

# face recognizer is trained
face_recognizer.save('face_trained.yml')
np.save('featurs.npy',featurs)
np.save('labels.npy',labels)




























########################### testing haar ##############################
# import cv2 as cv
#
# img = cv.imread('pictures/trainingSet/ed sheeran/images.jpg')
# img = cv.imread('pictures/testingSet/justin bieber/istockphoto-1300512246-170667a.jpg')
# #cv.imshow('ed',img)
#
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)
#
# haar_cascade = cv.CascadeClassifier('haar_face.xml')
# faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
#
# print(f'number of faces found = {len(faces_rect)}')
#
# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
#
# cv.imshow('faceDet',img)
# cv.waitKey(0)
