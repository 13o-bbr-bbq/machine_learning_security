## 7.0. 目的
本ブログは、**顔認証システムの実装**を通して、**Convolutional Neural Network**と呼ばれる**画像分類アルゴリズム**の使用方法を理解することを目的とします。

## 7.1. 顔認証
顔認証（Face Authentication）とは、対象人物の顔の特徴点を基に認証を行う機能であり、空港における入出国ゲートや部屋の入館管理、PCやスマートフォンへのログイン機構などに利用されています。  

顔認証では、カメラから取り込んだデジタル画像から人間の顔と思われる部分を抜き出し、顔画像データベースや顔画像モデルなどと照合を行います。照合する手法は様々であり、例えば、対象人物の目・鼻・口などの特徴点を顔画像データベースと照合する方法や、3次元センサで顔の立体的情報を取得し、眼窩（目のくぼみ）・鼻・顎の輪郭などと特徴を照合する方法、そして、皮膚のキメ（しわやしみ）を分析して本人確認を行う方法などもあります。  

### 7.1.1. コンセプト
本ブログでは、顔認証をDeep Learning技術で実装することにします。  

Deep Learningで顔認証を実装するためには、カメラから取り込んだ対象人物のデジタル画像から**顔部分を抜き出し**、顔の**特徴点を基に対象人物を分類**すれば良いと考えられます。分類のラベルは**認証成功** or **認証失敗**の**2クラス**とし、事前に学習済みの人物の場合は認証成功に分類、学習していない人物は認証失敗に分類します。  

また、分類時に**確率**も併せて計算できた方が運用上は好ましいと言えます。なぜならば、確率が求まれば「認証成功 or 認証失敗」の閾値を設けることができるためです。例えば、「99%以上の確率で認証成功」に分類された場合は認証成功とし、「99%未満の確率で認証成功」と分類された場合は認証失敗に分類します。このように閾値を設けることで、認証成功とすべきか判断に迷うケース（認証成功とすべき人物に顔が似ているが、確証が持てない場合など）の場合は、安全側に判断を倒して認証失敗に分類することができます（本例における閾値は99%）。  

本ブログではConvolutional Neural Networkを使用し、上記のコンセプトを持った顔認証システムを実装していきます。  

## 7.2. Convolutional Neural Network（CNN）
CNNは通常の**Neural NetworkにConvolution（畳み込み）を追加**したものであり、分類対象の画像に対する**高い頑健性（ロバスト性）**を持つという特徴があります。この特徴により、**高精度の画像分類**を実現します。  

### 7.2.1. CNN入門の入門
ところで、画像分類における頑健性（ロバスト性）とは何でしょうか？  
下記の猫画像で考えてみましょう。  

 <div align="center">
 <figure>
 <img src='./img/7-1_cat2.png' alt='cat1'><br>
 <figcaption>猫画像１</figcaption><br>
 <br>
 <br>
 <img src='./img/7-1_cat1.png' alt='cat2'><br>
 <figcaption>猫画像２</figcaption><br>
 </div>
 </figure>

猫画像１は**猫が横を向いている画像**です。一方、猫画像２は**猫が正面を向いている画像**です。この2つの画像はレイアウトが異なるものの、人間はどちらも猫として認識することができます。それは、猫は「耳が尖っている」「ピンと伸びたヒゲがある」「目がまん丸である」「鼻が三角である」といった**猫の特徴**を捉えているからであり、決して単純に認識対象の形状を重ね合わせて認識している訳ではありません。  

これと同じことを（Convolutionが無い）Neural Networkで実現しようとすると、ちょっと困った問題が発生します。Neural Networkは分類対象の画像を**1pixel単位**で受け取るため、分類対象画像のレイアウトが少しでも異なると、入力データの情報が大きく異なってしまいます。このため、上記の猫画像１と２のようにレイアウトが異なる場合、2つを同じ猫として認識することが難しくなります。つまり、頑健性が低いと言えます。  

 <center>
 <img src='./img/7-1_nn.png' alt='NNの説明'><br>
 <b>Neural Networkは頑健性が低い</b>
 </center>

一方CNNは、分類対象の画像を**Convolutionで畳み込み**を行って受け取るため、分類対象画像のレイアウトが多少異なる場合でも、その**差異を吸収**することができます。これにより、猫画像１と２のようにレイアウトが異なっている場合でも、同じ猫として認識することができます。つまり、頑健性が高いと言えます。  

 <center>
 <img src='./img/7-1_cnn.png' alt='CNNの説明(畳み込み、Poolingなど)'><br>
 <b>CNNの頑健性</b>
 </center>

畳み込みによりCNNは高い頑健性を誇るため、分類対象画像に対する柔軟性を持ち、それ故に高精度の画像分類を実現できます。なお、CNNで構築したモデルを実行すると、出力結果として**分類したクラス**と**分類確率**を得ることができます。また、CNNは教師あり学習であるため、分類を行う前に学習データ（教師データ）を用いて分類対象の特徴を学習させておく必要があります。  

以上でCNN入門の入門は終了です。  
次節では、CNNを使用した顔認証システムの構築手順とサンプルコードを解説します。

## 7.3. 顔認証システムの実装
本ブログで実装する顔認証システムは、大きく以下の4ステップで認証を行います。  

 1. 顔認識モデルの作成  
 2. Webカメラから顔画像を取得  
 3. 顔認識モデルで顔画像を分類  
 4. 分類確率と閾値を比較して認証成否を判定    

### 7.3.1. 顔認識モデルの作成
本ブログの顔認証システムは教師あり学習のCNNを使用します。  
よって、先ずは学習データを用意する必要があります。  

#### 7.3.1.1. 学習データの準備
学習データとして**認証対象とする人物の顔画像**を準備します。  
本ブログでは、一般公開されている**顔画像データセット「[VGGFACE2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)」から無作為に選んだ人物**と、**筆者**の計11人を認証対象とします。  

| VGGFACE2|
|:--------------------------|
| 大規模な顔画像データセット。Google画像検索からダウンロードされた**9,000人以上**の人物の顔画像が**330万枚以上**収録されており、年齢・性別・民族・職業・顔の向き・顔の明るさ（照明具合）など、バリエーション豊富である。|

以下に認証対象の人物を示します。  

 <center>
 <img src='.img/7-3_dataset.png'><br>
 <b>認証対象の人物</b>
 </center>

なお、当然ながら筆者の顔画像はVGGFACE2に含まれていないため、顔画像収集用のサンプルコード[`record_face.py`](src/chap7/record_face.py)を用いて収集します。このコードを実行すると、**0.5秒間隔**でWebカメラから画像を取り込み、顔を認識して顔部分のみを切り出します。  

```
your_root_path> python3 record_face.py
1/100 Capturing face image.
2/100 Capturing face image.
3/100 Capturing face image.
...snip...
98/100 Capturing face image.
99/100 Capturing face image.
100/100 Capturing face image.
```

 <center>
 <img src='./img/7-3_recording_face.png' width=500><br>
 <b>顔画像を収集している様子</b>
 </center>

そして、コード実行ディレクトリ配下に、顔画像を格納する「`original_image`」ディレクトリと、サブディレクトリ「`Isao-Takaesu`」を自動生成し、サブディレクトリに切り出した顔画像をJPEG形式で保存します。  

```
/your_root_path/original_image/Isao-Takaesu> ls
Isao-Takaesu_1.jpg
Isao-Takaesu_2.jpg
Isao-Takaesu_3.jpg
... snip ...
Isao-Takaesu_98.jpg
Isao-Takaesu_99.jpg
Isao-Takaesu_100.jpg
```

なお、サブディレクトリ名「`Isao-Takaesu`」が、**顔認証時のクラス名**となります。  
もし、あなたが任意の人物を認証対象としたい場合は、コード中の`FILE_NAME`を変更して本コードを実行してください。  

#### 7.3.1.2 データセットの作成
顔画像の収集が完了しましたので、学習データとモデル評価用のテストデータを作成します。  

`original_image`ディレクトリ配下に、VGGFACE2から選んだ10人の人物のサブディレクトリを作成します。  

```
your_root_path\original_image> mkdir Carolina-Kluft Carol-Swenson Cris-Judd Frank-de-Boer Junny-Lang-Ping John-Winchester Joumana-Kidd Leo-Sayer Zhang-Xin Western-Hagen

your_root_path\original_image> ls
Carolina-Kluft
Carol-Swenson
Cris-Judd
Frank-de-Boer
Isao-Takaesu
Junny-Lang-Ping
John-Winchester
Joumana-Kidd
Leo-Sayer
Zhang-Xin
Western-Hagen
```

各人の顔画像をVGGFACE2からコピーし、上記で作成した各人のサブディレクトリにペーストします。  
そして、データセットを生成するサンプルコード「[`create_dataset.py`](src/chap7/create_dataset.py)」を実行します。  

```
your_root_path> python3 create_dataset.py
```

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
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
            shutil.move(image, train_label_dir)
        else:
            shutil.move(image, test_label_dir)
```

このコードは、`original_image`配下の**認証対象人物のサブディレクトリ名をクラス名**とし、データセットのディレクトリ「`dataset`」配下に学習データを格納する「`train`」ディレクトリと、モデル評価用のテストデータを格納する「`test`」ディレクトリを自動生成します。そして、7対3の割合で認証対象人物の顔画像を`train`と`test`に振り分けます。  

#### 7.3.1.3 学習の実行
学習データの準備ができましたので、CNNを用いて認証対象人物の顔を学習します。  
学習を実行するサンプルコード「[`train.py`](src/chap7/train.py)」を実行します。  

```
your_root_path> python3 train.py
```

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

# dataset path.
dataset_path = os.path.join(full_path, 'dataset')

# Train/test data path.
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

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

# Save model.
model_path = os.path.join(full_path, 'model')
os.makedirs(model_path, exist_ok=True)
model_name = os.path.join(model_path, 'cnn_face_auth.h5')
print('Save model to {}'.format(model_name))
model.save_weights(model_name)
print('Finish training model.')
```

このコードは、`dataset/train`配下の学習データと`dataset/test`配下のテストデータを使用して顔認識モデルを作成します。作成されたモデルは、モデル格納用のディレクトリ「`model`」配下に`cnn_face_auth.h5`というファイル名で保存されます。  

### 7.3.2. サンプルコード及び実行結果
#### 7.3.2.1. サンプルコード
本ブログではPython3を使用し、簡易的な顔認証システムを実装しました。
※本コードは[こちら](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/src/chap7/face_auth.py)から入手できます。

本システムの大まかな処理フローは以下のとおりです。

 1. Webカメラから対象人物の映像を取得
 2. 取得した映像から顔部分の切り出し
 3. 切り出した顔画像を学習済みCNNモデルで分類
 4. 閾値を基に許容 or 拒否を判定

```
#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import configparser
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

# dataset path.
dataset_path = os.path.join(full_path, 'dataset')

# Model path.
model_path = os.path.join(full_path, 'model')
model_name = os.path.join(model_path, 'cnn_face_auth.h5')

MAX_RETRY = 50
THRESHOLD = 80.0

# Generate class list.
classes = os.listdir(test_path)
nb_classes = len(classes)

# Prepare model.
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

# Load model.
print('Load trained model: {}'.format(model_name))
model.load_weights(model_name)

# Use Loss=categorical_crossentropy.
print('Compile model.')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Execute face authentication.
capture = preparation.create_camera_instance()
for idx in range(MAX_RETRY):
    # Read 1 frame from VideoCapture.
    ret, image = capture.read()

    # Execute detecting face.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(os.path.join(full_path, 'haarcascade_frontalface_default.xml'))
    faces = cascade.detectMultiScale(gray_image,
                                     scaleFactor=1.1,
                                     minNeighbors=2,
                                     minSize=(128, 128))

    if len(faces) == 0:
        print('Face is not found.')
        continue

    for face in faces:
        # Extract face information.
        x, y, width, height = face
        predict_image = image[y:y + height, x:x + width]
        if predict_image.shape[0] < 128:
            continue
        predict_image = cv2.resize(predict_image, (128, 128))
        predict_image2 = cv2.resize(image, (128, 128))

        # Save image.
        file_name = os.path.join(dataset_path, 'tmp_face.jpg')
        cv2.imwrite(file_name, predict_image2)

        # Predict face.
        # Transform image to 4 dimension tensor.
        img = image.load_img(file_name, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Prediction.
        pred = model.predict(x)[0]
        top = 1
        top_indices = pred.argsort()[-top:][::-1]
        results = [(classes[i], pred[i]) for i in top_indices]

        judge = 'Reject'
        prob = results[0][1] * 100
        if prob > THRESHOLD:
            judge = 'Unlock'
        print('{} ({:.1f}%). res="{}"'.format(results[0][0], prob, judge))

        # Draw frame to face.
        cv2.rectangle(image,
                      (x, y),
                      (x + width, y + height),
                      (255, 255, 255),
                      thickness=2)

        # Get current date.
        date = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        print_date = datetime.strptime(date[:-3], '%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')

        # Display raw frame data.
        cv2.putText(image, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, print_date, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Face Authorization', image)

        file_name = os.path.join(dataset_path, 'tmp_face' + str(idx) + '_.jpg')
        cv2.imwrite(file_name, image)

    # Waiting for getting key input.
    k = cv2.waitKey(500)
    if k == 27:
        break

# Termination (release capture and close window).
capture.release()
cv2.destroyAllWindows()
```

#### 1.3.4.2. コード解説
今回はロジスティック回帰の実装に、機械学習ライブラリの**scikit-learn**を使用しました。
scikit-learnの使用方法は[公式ドキュメント](http://scikit-learn.org/)を参照のこと。

##### パッケージのインポート
```
from sklearn import linear_model
from sklearn import metrics
```

scikit-learnのロジスティック回帰パッケージ「`linear_model`」をインポートします。
このパッケージには、ロジスティック回帰を行うための様々なクラスが収録されています。

また、分類精度を評価するためのパッケージ「`metrics`」も併せてインポートします。

##### 学習データのロード
```
# Load train data
df_train = pd.read_csv('..\\dataset\\kddcup_train.csv')
X_train = df_train.iloc[:, [0, 7, 10, 11, 13, 35, 37, 39]]  # feature(X)
X_train = (X_train - X_train.mean()) / X_train.mean()       # normalization
y_train = df_train.iloc[:, [41]]                            # label(y)
```

学習データ「kddcup_train.csv」から、特徴選択した特徴量に紐付くデータとラベルを取得します（```X_train = df_train.iloc[:, [0, 7, 10, 11, 13, 35, 37, 39]]```、```y_train = df_train.iloc[:, [41]]```）。
ここで、学習精度を上げるために、各特徴量のデータ値を**正規化**します（```(X_train - X_train.mean()) / X_train.mean()```）。

| 正規化（Normalization）|
|:--------------------------|
| 尺度が異なるデータ値を```0～1```等の一定の範囲に収まるように加工し、各特徴量の尺度を統一する手法。分類結果が数値的に大きな特徴量に大きく左右されることを防ぐ。正規化することで、学習精度が向上する場合がある。|

##### テストデータのロード
```
# Load test data
df_test = pd.read_csv('..\\dataset\\kddcup_test.csv')
X_test = df_test.iloc[:, [0, 7, 10, 11, 13, 35, 37, 39]]
X_test = (X_test - X_test.mean()) / X_test.mean()
y_test = df_test.iloc[:, [41]]
```

テストデータを学習データと同様に取得します（`X_test`、`y_test`）。

##### モデルの定義
```
logreg = linear_model.LogisticRegression(C=1e5)
```

`linear_model.LogisticRegression`でロジスティック回帰モデルを定義します。
なお、引数として渡している「`C=1e5`」は**正則化**強度です。

| 正則化（Regularization）|
|:--------------------------|
| 機械学習において過学習を抑えるための手法。|

##### モデルの作成（学習の実行）
```
logreg.fit(X_train, y_train)
```

`logreg`の`fit`メソッドの引数として、各特徴量「`X_train`」とラベル「`y_train`」を渡すことで、学習が行われます。これにより、ロジスティック回帰モデルが作成されます。

##### 分類確率の取得
```
probs = logreg.predict_proba(X_test)
```

モデル`logreg`の`predict_proba`メソッドの引数としてテストデータ「`X_test`」を渡すことで、モデルがテストデータの分類を行い、分類確率「`probs`」を返します。

##### 分類結果の取得
```
y_pred = logreg.predict(X_test)
```

モデル`logreg`の`predict`メソッドの引数としてテストデータ「`X_test`」を渡すことで、モデルがテストデータの分類を行い、分類結果「`y_pred`」を返します。

##### モデルの評価
```
print('score : {0}'.format(metrics.accuracy_score(y_test, y_pred)))
```

モデルが分類した結果「`y_pred`」と、予め用意したラベル「`y_test`」を`metrics.accuracy_score`メソッドの引数として渡すことで、モデルの分類結果とラベルの一致度合（分類精度）を計測することができます。

##### 分類結果の出力
```
for predict, prob in zip(y_pred, probs):
    print('{0}\t{1}\t{2}'.format(y_test.iloc[idx, [0]].values[0], predict, np.max(prob)))
    idx += 1
```

「`y_test`」「`y_pred`」「`probs`」を1件ずつコンソール上に出力します。

#### 1.3.4.3. 実行結果
このサンプルコードを実行すると、以下の結果がコンソール上に出力されます。

```
train_time   : 10.235417526819267 [sec]
predict_time : 0.023557101303234518 [sec]
score : 0.8996588708933895
------------------------------------------------------------
label             predict         probability
normal.           normal.         0.9998065736764143
normal.           normal.         0.9998065736764143
normal.           normal.         0.9998065736764143
normal.           normal.         0.9917140229898096

...snip...

nmap.             nmap.           0.695479203
nmap.             nmap.           0.695479203
nmap.             nmap.           0.695479203
nmap.             nmap.           0.695479203

...snip...

buffer_overflow.  normal.         0.999980104
buffer_overflow.  guess_passwd.   0.333345516
buffer_overflow.  normal.         0.990855252
buffer_overflow.  guess_passwd.   0.333334087

...snip...

teardrop.         teardrop.       0.999998379
teardrop.         teardrop.       0.500005699
teardrop.         teardrop.       0.500000044
teardrop.         teardrop.       0.999998157

...snip...

guess_passwd.     guess_passwd.   0.478403815
guess_passwd.     guess_passwd.   0.499999391
guess_passwd.     guess_passwd.   0.38504377
guess_passwd.     normal.         0.999790317
```

`score`はモデルの分類精度であり、分類精度は「89.9%」であることを示しています。
`-----`以下は、通信データ毎の「ラベル、分類結果、分類確率」を出力しています。

この結果から、「normal」や「nmap」「teardrop」「guess_passwd」は概ね正しく分類できていることが分かります。但し、「nmap」や「teardrop」「guess_passwd」の一部分類確率を見ると、「69%、50%、38%」等のように低い数値になっています。これは、今回筆者が主観で選択した特徴量よりも、**もっと適切な特徴量が存在する可能性**を示唆しています。また、「buffer_overflow」は殆ど正しく分類できておらず、「normal」や「guess_passwd」として誤検知していることが分かります。これは、「buffer_overflow」の**特徴量を見直す必要がある**ことを強く示唆しています。

## 1.4. おわりに
このように、特徴選択の精度によって分類精度が大きく左右されることが分かって頂けたかと思います（笑）。
 本ブログでは、KDD Cup 1999のデータセットを学習データとして使用しましたが、実際に収集したリアルなデータを学習に使用する場合は、侵入検知に**役立つ特徴量を見落とさないように、収集する情報を慎重に検討**する必要があります（ここがエンジニアの腕の見せ所だと思います）。

本ブログを読んでロジスティック回帰にご興味を持たれた方は、ご自身で特徴選択を行ってコードを実行し、検知精度がどのように変化するのかを確認しながら理解を深める事を推奨致します。

## 1.5. 動作条件
 * Python 3.6.1
 * pandas 0.23.4
 * numpy 1.15.1
 * scikit-learn 0.19.2

### CHANGE LOG
* 2018.09.25 : 初稿
