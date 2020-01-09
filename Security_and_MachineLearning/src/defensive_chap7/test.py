#!/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

# dataset path.
dataset_path = os.path.join(full_path, 'dataset')
test_path = os.path.join(dataset_path, 'test')

# Model path.
model_path = os.path.join(full_path, 'model')
trained_model = os.path.join(model_path, 'cnn_face_auth.h5')

MAX_RETRY = 50
THRESHOLD = 80.0

# Generate class list.
classes = os.listdir(test_path)
nb_classes = len(classes)

# Dimensions of training images.
img_width, img_height = 150, 150

# Load model.
print('Load trained model: {}'.format(trained_model))
model = load_model(trained_model)

# Separate images to train and test.
for label in classes:
    # Target test label.
    test_label_path = os.path.join(test_path, label)

    # Get images of label.
    in_image = glob.glob(os.path.join(test_label_path, label + '*' + '.jpg'))
    for idx, test_image in enumerate(in_image):
        # Transform image to 4 dimension tensor.
        img = image.img_to_array(image.load_img(test_image, target_size=(img_width, img_height)))
        x = np.expand_dims(img, axis=0)
        x = x / 255.0

        # Prediction.
        preds = model.predict(x)[0]
        predict_label = np.argmax(preds)
        if label == classes[predict_label]:
            print('{}\tOK\tReal: {}, Predict: {}/{:.2f}%'.format(os.path.basename(test_image),
                                                                 label,
                                                                 classes[predict_label],
                                                                 preds[predict_label]*100))
        else:
            print('{}\tNG\tReal: {}, Predict: {}/{:.2f}%'.format(os.path.basename(test_image),
                                                                 label,
                                                                 classes[predict_label],
                                                                 preds[predict_label] * 100))
