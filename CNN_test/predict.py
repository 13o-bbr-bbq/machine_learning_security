# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing import image

# CIFAR10のクラス一覧
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
nb_classes = len(classes)

# CIFAR10の画像情報(x:32pic, y:32pic, channel:3)
img_height, img_width = 32, 32
channels = 3

# VGG16
input_tensor = Input(shape=(img_height, img_width, channels))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC
fc = Sequential()
fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
fc.add(Dense(256, activation='relu'))
fc.add(Dropout(0.5))
fc.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=fc(vgg16.output))

# 学習済みの重み(finetuning.pyでFine-tuningした重み)をロード
model.load_weights(os.path.join('cifar10\\results', 'finetuning.h5'))
# model.load_weights(os.path.join('cifar10\\results', 'finetuning_noise.h5'))

# 多クラス(=10)分類なのでloss=categorical_crossentropyとする
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if sys.argv[1] == '':
    print("usage: Test file is not found.")
    sys.exit(1)

# 画像を4次元テンソルに変換
img = image.load_img(sys.argv[1], target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# クラスの予測(top3を出力)
pred = model.predict(x)[0]
top = 3
top_indices = pred.argsort()[-top:][::-1]
result = [(classes[i], pred[i]) for i in top_indices]
for x in result:
    print(x)
