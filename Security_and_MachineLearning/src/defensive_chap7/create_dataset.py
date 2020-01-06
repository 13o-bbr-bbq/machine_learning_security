#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import cv2
import glob

# Separation rate of target images to Train/Test.
SEP_RATE = 0.7

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

# Original image path.
original_image_path = os.path.join(full_path, 'original_image')

# Create dataset path.
dataset_path = os.path.join(full_path, 'dataset')
os.makedirs(dataset_path, exist_ok=True)

# Create train/test path.
train_path = os.path.join(dataset_path, 'train')
os.makedirs(train_path, exist_ok=True)
test_path = os.path.join(dataset_path, 'test')
os.makedirs(test_path, exist_ok=True)

# Execute face recognition in saved image.
label_list = os.listdir(original_image_path)
for label in label_list:
    # Extract target image each label.
    target_dir = os.path.join(original_image_path, label)
    in_image = glob.glob(os.path.join(target_dir, '*'))

    # Detect face in image.
    for idx, image in enumerate(in_image):
        # Read image to OpenCV.
        cv_image = cv2.imread(image)

        # If image size is smaller than 128, it is excluded.
        if cv_image.shape[0] < 128:
            print('This face is too small: {} pixel.'.format(str(cv_image.shape[0])))
            continue
        save_image = cv2.resize(cv_image, (128, 128))

        # Save image.
        file_name = os.path.join(dataset_path, label + '_' + str(idx+1) + '.jpg')
        cv2.imwrite(file_name, save_image)

# Separate images to train and test.
for label in label_list:
    # Define train directory each label.
    train_label_path = os.path.join(train_path, label)
    os.makedirs(train_label_path, exist_ok=True)

    # Define test directory each label.
    test_label_path = os.path.join(test_path, label)
    os.makedirs(test_label_path, exist_ok=True)

    # Get images of label.
    in_image = glob.glob(os.path.join(dataset_path, label + '*' + '.jpg'))
    for idx, image in enumerate(in_image):
        if idx < len(in_image) * SEP_RATE:
            shutil.move(image, train_label_path)
        else:
            shutil.move(image, test_label_path)
