#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

# Dataset path.
dataset_path = os.path.join(full_path, 'dataset')

# Train/test data path.
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

# Model path.
model_path = os.path.join(full_path, 'model')
os.makedirs(model_path, exist_ok=True)
model_weights = os.path.join(model_path, 'cnn_face_auth.h5')
model_arch = os.path.join(model_path, 'cnn_face_auth.json')

# Generate class list.
classes = os.listdir(test_path)
nb_classes = len(classes)

print('Start training model.')

# Build VGG16.
print('Build VGG16 model.')
input_tensor = Input(shape=(128, 128, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# Build FC.
print('Build FC model.')
fc = Sequential()
fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
fc.add(Dense(256, activation='relu'))
fc.add(Dropout(0.5))
fc.add(Dense(nb_classes, activation='softmax'))

# Connect VGG16 and FC.
print('Connect VGG16 and FC.')
model = Model(input=vgg16.input, output=fc(vgg16.output))

# Freeze before last layer.
for layer in model.layers[:15]:
    layer.trainable = False

# Use Loss=categorical_crossentropy.
print('Compile model.')
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# Generate train/test data.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=32,
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=32,
    shuffle=True)

# Count train and test data.
# Count train data.
train_count = 0
for root, dirs, files in os.walk(train_path):
    train_count += len(files)

# Count test data.
test_count = 0
for root, dirs, files in os.walk(test_path):
    test_count += len(files)

# Fine-tuning.
print('Execute fine-tuning.')
history = model.fit_generator(
    train_generator,
    samples_per_epoch=train_count,
    nb_epoch=100,
    validation_data=validation_generator,
    nb_val_samples=test_count)

# Save model (weights and architecture).
with open(model_arch, 'w') as fout:
    json.dump(model.to_json(), fout)
    print('Save model archtecture to {}'.format(model_arch))
print('Save model weights to {}'.format(model_weights))
model.save_weights(model_weights)

print('Finish training model.')
