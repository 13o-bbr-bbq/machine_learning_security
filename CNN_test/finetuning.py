# -*- coding: utf-8 -*-
import os
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras import optimizers


# CIFAR10のクラス一覧
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

batch_size = 32
nb_classes = len(classes)

# CIFAR10の画像情報(x:32pic, y:32pic, channel:3)
img_rows, img_cols = 32, 32
channels = 3

# 訓練用画像とテスト用画像のPath
train_data_dir = '.\\cifar10\\train_image'
validation_data_dir = '.\\cifar10\\test_image'

# 訓練用画像(各クラス5000枚)、テスト用画像(各クラス1000枚)
# epochは30くらいでも十分か？
nb_train_samples = 50000
nb_val_samples = 10000
nb_epoch = 50

# 学習済み重みの保存Path
result_dir = '.\\cifar10\\results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


if __name__ == '__main__':
    # VGG16モデルと学習済み重み(ImageNet)をロード
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFCを接続
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 最後の畳み込み層の手前までFreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    # 多クラス(=10)分類なのでloss=categorical_crossentropyとする
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    # Fine-tuning
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples)

    model.save_weights(os.path.join(result_dir, 'finetuning.h5'))
