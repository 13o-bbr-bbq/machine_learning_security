#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2

FILE_NAME = 'Isao-Takaesu'
CAPTURE_NUM = 100

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

# Dimensions of captured images.
img_width, img_height = 128, 128

# Set web camera.
capture = cv2.VideoCapture(0)

# Create saved base Path.
saved_base_path = os.path.join(full_path, 'original_image')
os.makedirs(saved_base_path, exist_ok=True)

# Create saved path of your face images.
saved_your_path = os.path.join(saved_base_path, FILE_NAME)
os.makedirs(saved_your_path, exist_ok=True)

for idx in range(CAPTURE_NUM):
    # Read 1 frame from VideoCapture.
    print('{}/{} Capturing face image.'.format(idx + 1, CAPTURE_NUM))
    ret, image = capture.read()

    # Execute detecting face.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(os.path.join(full_path, 'haarcascade_frontalface_default.xml'))
    faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2, minSize=(img_width, img_height))

    if len(faces) == 0:
        print('Face is not found.')
        continue

    for face in faces:
        # Extract face information.
        x, y, width, height = face
        face_size = image[y:y + height, x:x + width]
        if face_size.shape[0] < img_width:
            print('This face is too small: {} pixel.'.format(str(face_size.shape[0])))
            continue

        # Save image.
        file_name = FILE_NAME + '_' + str(idx+1) + '.jpg'
        save_image = cv2.resize(face_size, (img_width, img_height))
        cv2.imwrite(os.path.join(saved_your_path, file_name), save_image)

        # Display raw frame data.
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), thickness=2)

        # Display raw frame data.
        msg = 'Captured {}/{}.'.format(idx + 1, CAPTURE_NUM)
        cv2.putText(image, msg, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Captured your face', image)

    # Waiting for getting key input.
    k = cv2.waitKey(500)
    if k == 27:
        break

# Termination (release capture and close window).
capture.release()
cv2.destroyAllWindows()
