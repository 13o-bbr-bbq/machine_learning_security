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

 ![cat1](./img/7-1_cat2.png)  
 猫画像１  

 ![cat2](./img/7-1_cat1.png)  
 猫画像２  

猫画像１は**猫が横を向いている画像**です。一方、猫画像２は**猫が正面を向いている画像**です。この2つの画像はレイアウトが異なるものの、人間はどちらも猫として認識することができます。それは、猫は「耳が尖っている」「ピンと伸びたヒゲがある」「目がまん丸である」「鼻が三角である」といった**猫の特徴**を捉えているからであり、決して単純に認識対象の形状を重ね合わせて認識している訳ではありません。  

これと同じことを（Convolutionが無い）Neural Networkで実現しようとすると、ちょっと困った問題が発生します。Neural Networkは分類対象の画像を**1pixel単位**で受け取るため、分類対象画像のレイアウトが少しでも異なると、入力データの情報が大きく異なってしまいます。このため、上記の猫画像１と２のようにレイアウトが異なる場合、2つを同じ猫として認識することが難しくなります。つまり、頑健性が低いと言えます。  

 ![NNの説明](./img/7-1_nn.png)  

一方CNNは、分類対象の画像を**Convolutionで畳み込み**を行って受け取るため、分類対象画像のレイアウトが多少異なる場合でも、その**差異を吸収**することができます。これにより、猫画像１と２のようにレイアウトが異なっている場合でも、同じ猫として認識することができます。つまり、頑健性が高いと言えます。  

 ![CNNの説明（畳み込み、Poolingなど）](./img/7-1_cnn.png)  

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
身近な人の顔画像を晒すのはプライバシー侵害となるため、本ブログでは一般公開されている顔画像データセット「[VGGFACE2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)」を使用することにします。  

| VGGFACE2|
|:--------------------------|
| 大規模な顔画像データセット。Google画像検索からダウンロードされた**9,000人以上**の人物の顔画像が**330万枚以上**収録されており、年齢・性別・民族・職業・顔の向き・顔の明るさ（照明具合）など、バリエーション豊富である。|

以下に、顔認証システムで認証対象とする人物の一例を示します。  

 ![学習データ](./img/7-3_dataset.png)  
 学習データの一例（≒認証する人物）  

11人中10人はVGGFACE2に収録されている人物とし、これに筆者「Isao Takaesu」を加えています。  
当然ながら、筆者の顔画像はVGGFACE2に含まれていないため、ラップトップPCのWebカメラで自撮りして収集しました。  

以下に、Webカメラで画像を収集するサンプルコードを記します。  

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2

FILE_NAME = 'Isao-Takaesu'
CAPTURE_NUM = 100

# Full path of this code.
full_path = os.path.dirname(os.path.abspath(__file__))

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
    print('{}/{} Capturing face image.'.format(idx + 1, 10))
    ret, image = capture.read()

    # Execute detecting face.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(os.path.join(full_path, 'haarcascade_frontalface_default.xml'))
    faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2, minSize=(128, 128))

    if len(faces) == 0:
        print('Face is not found.')
        continue

    for face in faces:
        # Extract face information.
        x, y, width, height = face
        face_size = image[y:y + height, x:x + width]
        if face_size.shape[0] < 128:
            print('This face is too small: {} pixel.'.format(str(face_size.shape[0])))
            continue

        # Save image.
        file_name = FILE_NAME + '_' + str(idx) + '.jpg'
        save_image = cv2.resize(face_size, (128, 128))
        cv2.imwrite(os.path.join(saved_your_path, file_name), save_image)

        # Display raw frame data.
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), thickness=2)

        # Display raw frame data.
        msg = 'Captured {}/{}.'.format(idx + 1, CAPTURE_NUM)
        cv2.putText(image, msg, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Captured your face', image)

    # Waiting for getting key input.
    k = cv2.waitKey(1)
    if k == 27:
        break

# Termination (release capture and close window).
capture.release()
cv2.destroyAllWindows()
```

このコードは1秒間隔でWebカメラ経由で画像を取り込み、画像に人物が含まれる場合は顔部分を切り出してファイルに保存します。  
本コードを実行すると、以下のようにWebカメラに映る人物の顔画像を収集することができます。  

 ![切り出した筆者の顔画像](./img/7-3_clipped_face.png)   
 収集した筆者の顔画像  

なお、ファイル名を変更したい場合は`FILE_NAME`、収集する枚数を変更したい場合は `CAPTURE_NUM`の値を適宜変更してください。  

もし、あなたが任意の人物を認証対象としたい場合は、上記のサンプルコードを使用することで、対象人物の学習データを作成することができます。  

#### 7.3.1.2 データセットの作成


#### 7.3.1.3 学習の実行
学習データの準備ができましたので、CNNで顔を学習します。  
本ブログでは、CNNの実装にKeras（バックエンドはTensorflow）を使用しました。以下に、学習を行うサンプルコードを記します。  

```
Sample codes.
```

このコードでは、

### 1.3.4. サンプルコード及び実行結果
#### 1.3.4.1. サンプルコード
本ブログではPython3を使用し、簡易的な侵入検知システムを実装しました。
※本コードは[こちら](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/src/intrusion_detection.py)から入手できます。

本システムの大まかな処理フローは以下のとおりです。

 1. 学習データ及びテストデータのロード
 2. 学習データを使用して学習（モデル作成）
 3. Webカメラから対象人物の映像を取得
 4. 取得した映像から顔部分をクリッピング
 5. クリッピングした顔画像をモデルで分類
 6. 許容 or 拒否の判定

```
#!/bin/env python
# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics

# Load train data
df_train = pd.read_csv('..\\dataset\\kddcup_train.csv')
X_train = df_train.iloc[:, [0, 7, 10, 11, 13, 35, 37, 39]]  # feature(X)
X_train = (X_train - X_train.mean()) / X_train.mean()       # normalization
y_train = df_train.iloc[:, [41]]                            # label(y)

# Load test data
df_test = pd.read_csv('..\\dataset\\kddcup_test.csv')
X_test = df_test.iloc[:, [0, 7, 10, 11, 13, 35, 37, 39]]
X_test = (X_test - X_test.mean()) / X_test.mean()
y_test = df_test.iloc[:, [41]]

# We create an instance of Logistic Regression Classifier.
logreg = linear_model.LogisticRegression(C=1e5)

# Fit (Train)
start = time.perf_counter()
logreg.fit(X_train, y_train)
elapsed_time = time.perf_counter() - start
print('train_time   : {0}'.format(elapsed_time) + ' [sec]')

# Predict for probability and label
start = time.perf_counter()
probs = logreg.predict_proba(X_test)
elapsed_time = time.perf_counter() - start
print('predict_time : {0}'.format(elapsed_time) + ' [sec]')
y_pred = logreg.predict(X_test)

# Evaluation
print('score : {0}'.format(metrics.accuracy_score(y_test, y_pred)))

# Output predict result
print('-' * 60)
print('label\tpredict\tprobability')
idx = 0
for predict, prob in zip(y_pred, probs):
    print('{0}\t{1}\t{2}'.format(y_test.iloc[idx, [0]].values[0], predict, np.max(prob)))
    idx += 1

print('finish!!')
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
