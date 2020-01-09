#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.preprocessing import image

# Check argument.
if len(sys.argv) != 2:
    print('Test image not found.')
    sys.exit(0)

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

# dataset path.
dataset_path = os.path.join(full_path, 'dataset')
test_path = os.path.join(dataset_path, 'test')

# Model path.
model_path = os.path.join(full_path, 'model')
trained_model = os.path.join(model_path, 'cnn_face_auth.h5')

# Generate class list.
classes = os.listdir(test_path)

# Dimensions of test images.
img_width, img_height = 128, 128

# Load model.
print('Load trained model: {}'.format(trained_model))
model = load_model(trained_model)

# Transform image to 4 dimension tensor.
img = image.img_to_array(image.load_img(sys.argv[1], target_size=(img_width, img_height)))
x = np.expand_dims(img, axis=0)
x = x / 255.0

# Prediction.
preds = model.predict(x)[0]
predict_label = np.argmax(preds)
print('Predict: {}/{}%'.format(classes[predict_label], preds[predict_label]))


# Visualization Conv layers.
def plot_conv_outputs(layer_num, layer_name, outputs):
    filters = outputs.shape[2]
    for idx in range(filters):
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.subplot(filters/6 + 1, 6, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('filter {}'.format(idx))
        plt.imshow(outputs[:, :, idx])
    plt.savefig('{}_{}.jpg'.format(layer_name, layer_num))


# Create a conv model.
conv_layers = [l.output for l in model.layers]
conv_model = Model(inputs=model.inputs, outputs=conv_layers)

# Prediction using conv model.
conv_outputs = conv_model.predict(x)

for i in range(len(conv_outputs)):
    print(f'layer {i}:{conv_outputs[i].shape}')

# Show all layers.
for idx in range(len(conv_outputs)):
    print(conv_layers[idx].name)
    if 'conv2d' in conv_layers[idx].name or 'max_pooling2d' in conv_layers[idx].name:
        plot_conv_outputs(idx, conv_layers[idx].name.split('/')[0], conv_outputs[idx][0])

print('Finish.')
