#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


# Count samples.
def sample_count(sample_path):
    sample = 0
    for root, dirs, files in os.walk(sample_path):
        sample += len(files)
    return sample


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
trained_model_path = os.path.join(model_path, 'cnn_face_auth.h5')

# Generate class list.
classes = os.listdir(test_path)
nb_classes = len(classes)

# Dimensions of training images.
img_width, img_height = 128, 128

# Hyper parameters.
epochs = 100
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print('Start training model.')

# Build CNN architecture.
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# Use Loss=categorical_crossentropy.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate train data.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

# Generate test data.
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

# Count train/test samples.
train_samples = sample_count(train_path)
test_samples = sample_count(test_path)
print('train samples: {}\ntest samples: {}'.format(train_samples, test_samples))

# Training.
print('Execute training..')
model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=test_samples // batch_size)

# Save trained model.
print('Save model weights to {}'.format(trained_model_path))
model.save(trained_model_path)

print('Finish.')
