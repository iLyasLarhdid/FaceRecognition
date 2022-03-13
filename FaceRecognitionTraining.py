import os
import numpy as np
import cv2 as cv

people = []
for i in os.listdir(r'E:\ISGA\python\OpenCv\faceDetection\pictures\trainingSet'):
    people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
DIR = r'E:\ISGA\python\OpenCv\faceDetection\pictures\trainingSet'

features = []
labels = []


# loop in every folder and grab faces and
# add them into the training set which has features and corresponding label
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            for (x, y, w, h) in faces_rect:
                # face ragion of intrest
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print(f'length of the featurs list is : {len(features)}, labels = {len(labels)}')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

print('training ---------------------------------100% finished')
features = np.array(features, dtype='object')
labels = np.array(labels)
# training on features and labels

face_recognizer.train(features, labels)

# face recognizer is trained
face_recognizer.save('face_trained.yml')
np.save('featurs.npy', features)
np.save('labels.npy', labels)
