# https://realpython.com/traditional-face-detection-python/

import cv2 
import os
#os.chdir('Desktop/')
# Read image from your local file system
original_image = cv2.imread('modi-cabinet.jpg')
#original_image = cv.imread('/Desktop/0.jpg')
# Convert color image to grayscale for Viola-Jones
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
#face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt1.xml')

detected_faces = face_cascade.detectMultiScale(grayscale_image)

for (column, row, width, height) in detected_faces:
    cv2.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
    
cv2.imshow('Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
