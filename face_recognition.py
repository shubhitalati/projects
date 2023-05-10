import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_f.xml')
people = ['miley','taylor','selena','shawn']
#features = np.load('features.npy',allow_pickle=True)
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'D:\Users\Shubhi\OneDrive\Desktop\shubhi123\face_datas\selena\images (5).jpeg')

# Changes made by Sam 
if img is None:
    print("Error: Could not open/read image file")
    exit()
# change 1

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('person',gray)
# cv.waitKey(0) #Changes made by Sam

# faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))


for(x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]
    faces_roi = cv.resize(faces_roi, (100, 100)) #Changes made by Sam

    label,confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0), thickness=2)

    cv.imshow('detected face',img)
    cv.waitKey(0)