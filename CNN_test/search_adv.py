# -*- coding: utf-8 -*-
import os
import random
import datetime
import numpy as np
import cv2
from progressbar import ProgressBar
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing import image

# 検証対象のクラス
CLASS = 'cat'
TEST_DIR = 'cifar10\\test_image\\' + CLASS + '\\'
ADV_DIR = 'results\\' + CLASS + '\\'
TEST_IMAGES = 1000


# ランダム選択したピクセルに細工を加える
def random_adv(perturbation_filename, p):
    # 画像読み込み
    img = cv2.imread(TEST_DIR + perturbation_filename, cv2.IMREAD_UNCHANGED)

    # 画像サイズの取得
    if len(img) == 3:
        height, width, channel = img.shape[:3]
    else:
        height, width = img.shape[:2]

    for i in range(p):
        # 細工するピクセルをランダムに選択
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # ピクセル値の細工
        pixel = img[y, x]
        average = sum(pixel) / len(pixel)

        if average < 128:
            img[y, x] = [0, 0, 0]
        else:
            img[y, x] = [255, 255, 255]

    # 細工画像の保存
    adv_filename = 'adv_' + filename
    cv2.imwrite(ADV_DIR + adv_filename, img)

    return adv_filename


# クラスの予測
def predict(target_filename, height, width):
    # 画像を4次元テンソルに変換
    img = image.load_img(target_filename, target_size=(height, width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # クラスの予測
    pred = model.predict(x)[0]
    top = 1
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]

    '''
    result[0][0] : predict label
    result[0][1] : prediction probability
    '''
    return result

if __name__ == "__main__":
    # CIFAR10のクラス一覧
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    nb_classes = len(classes)

    # CIFAR10の画像情報(x:32pixel, y:32pixel, channel:3)
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

    # 細工するピクセル数(p)、1画像たりの細工試行最大回数(p_max_num)
    p = 1
    p_max_num = 100

    # 検証対象の画像一覧
    test_files = os.listdir(TEST_DIR)

    # 検証結果保存用のファイル
    today_detail = datetime.datetime.today()
    test_result = str(today_detail.strftime("%Y%m%d_%H%M%S")) + '.txt'
    f = open(test_result, 'w')
    f.write('Idx\tnormal image\tadversarial image\tmisclassify\tpredict\n')

    # 細工処理
    count = 0
    progress_count = 1
    progress = ProgressBar(min_value=1, max_value=TEST_IMAGES)
    for filename in test_files:
        # 進捗表示
        progress.update(progress_count)
        progress_count += 1

        # 細工前画像のクラス分類予測
        result = predict(TEST_DIR + filename, img_height, img_width)

        # 正しく分類できない画像は検証しない
        if CLASS != result[0][0]:
            continue

        for i in range(p_max_num):
            p_filename = random_adv(filename, p)

            # 細工画像のクラス分類予測
            result = predict(ADV_DIR + p_filename, img_height, img_width)

            # 誤分類した場合はログに書き込んで次の画像の検証を行う
            if CLASS != result[0][0]:
                count += 1
                f.write(str(count)+'\t'+filename+'\t'+p_filename+'\t'+result[0][0]+'\t'+str(result[0][1])+'\n')
                break
    print('\n')
    f.close()
