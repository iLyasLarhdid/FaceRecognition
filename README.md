# Face recognition code you can use in 3 steps

***first*** : 
>create folders inside the `./pictures/trainingSet` and `./pictures/testingSet` folders with the **names** of people you want to train the AI to recognize

***second*** :
>in `FaceRecognitionTraining.py` on line 5 change the DIR variable to your path, then execute the file

***third*** : 
>in the code of FaceRecognition.py change line 16 to img = cv.imread(r'`{full path}`\pictures\testingSet\ `{the name of one of the folders you created}` \ `{one of the images on that folder}`.jpg')

### make sure to install these packages
- `pip install opencv-contrib-python`
- `pip install numpy`
## Enjoy!