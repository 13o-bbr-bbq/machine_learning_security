## 2. スパム検知
　スパム検知（Spam Detection）とは、受信者の意向を無視して、無差別かつ大量にばら撒かれるスパムメールを検知する機能です。メジャーなメーラーには標準で実装されています。  

　本ブログでは、**ナイーブベイズ（Naive Bayes）**と呼ばれる手法で**スパム検知システム**を実現する例を示します。  

　以下、ナイーブベイズの簡単な解説を行った後、Pythonによる実装コードの例示と実行結果および考察を示していきます。  

### 2.1. ナイーブベイズ
　ナイーブベイズは**テキスト分類**を行う機械学習アルゴリズムであり、予め定義したカテゴリ（=クラス）に対象のテキストを自動分類することができます。このナイーブベイズは、スパム検知の他に、ブログサイトにおける投稿されたブログの自動カテゴライズ（政治、スポーツ、科学技術、エンタメ等）や[Webページの種別判定](https://www.mbsd.jp/blog/20160113.html)等に使用されています。実装が容易で、かつ処理速度も速いため、様々な用途に広く使われています。  

　次節では、ナイーブベイズを使用したスパム検知システムの構築手順と実装コードを解説します。  

### 2.2. ナイーブベイズを用いたスパム検知システム
　本ブログでは、受信したメールの本文内容を分析し、メールがスパムなのか正常なのかを検知するシステムを構築します。このような振る舞いをナイーブベイズで実現するためには、事前にスパムメールと正常メールの特徴をシステムに学習させたナイーブベイズモデルを作成します。そして、新たににメールを受信した場合、モデルにメール本文のテキストを与え、学習に基づいてメールがスパムなのか正常なのか分類（予測）させます。  

　以下、モデルの構築手順を順を追って解説します。  

#### 2.2.1. 学習データの準備
　ナイーブベイズは教師あり学習のため、事前にスパムや正常メールを含んだ学習データを準備する必要があります。  
　本来であれば、筆者が実際に受信したスパムメールを使用することが好ましいですが、不幸なことにサンプル数が少ないため、スパムメール事例を掲載した下記サイトからスパムメールを収集します。  

 * [最近の迷惑メール例](http://zero-net.jugem.jp/?eid=791)  
 * [つい読んでしまいたくなる？本当にあった迷惑（スパム）メール例集](https://securitynavi.jp/1081)  

　これらのサイトには様々なスパムメールの事例が掲載されていることから、スパム検知システムの学習データとして使用することにしました。  
　なお、正常メールについては、筆者が信頼できる相手から受信したメールをサンプリングします。  

　次に、スパムメールと正常メールを特徴付ける単語を選別します。  
　後述しますが、今回はメール本文から単語を切り出すために`nltk`（Natural Language Toolkit）を使用します。よって、収集したスパムメールと正常メールの本文を`nltk`を使用して**単語分割**します。  

| 単語分割|
|:--------------------------|
| 日本語や中国語などのように、単語間に区切りが存在しない膠着語において、単語の間に明示的な区切りを入れる**自然言語処理**の手法。なお、英語は単語間に空白が存在するため、容易に単語分割が可能。|

| 自然言語処理（Natural Language Processing）|
|:--------------------------|
| 人間が日常的に使用している自然言語をコンピュータで処理する技術群。|

　分割された単語の中から、スパムメールおよび正常メールを特徴付けると思われる単語を選別し、それぞれ**Spam**/**Normal**というようにラベル付けを行います（ラベル名は任意）。すると、以下のようなファイル内容になります。  

```
有料,Spam
無料,Spam
料金,Spam
特典付,Spam
懸賞付,Spam
登録料金,Spam
利用料金,Spam
料金未払,Spam
未払い金,Spam
支払い,Spam
最終通告,Spam
本通知,Spam
連絡,Spam
配信停止,Spam
再開,Spam

...snip...

確認,Normal
修正,Normal
アンケート,Normal
機能,Normal
結果,Normal
質問事項,Normal
診断日程延期,Normal
相談,Normal
連絡,Normal
メディア,Normal
記事掲載,Normal
本件,Normal
当社,Normal
Web,Normal
サイト,Normal
```

　1カラム目は特徴量、2カラム目がラベル（メールの種別＝答え）を表しています。  
　この状態のファイルを学習データ「[train_spam.csv](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/dataset/train_spam.csv)」とし、**UTF-8**で保存しておきます。  

#### 2.2.2. テストデータの準備
　スパム検知システムの性能を評価するためのテストデータを準備します。  
　ここでは、自分がスパムメールの送信者になったつもりで、自由に怪しげなメールを作成します。  

 * スパムメール例  
 ```
 件名：警告
 これは最終通告です。
 5分以内に以下のサイトにアクセスして、退会手続を行ってください。
 http://www.example.com

 退会しない場合は、即刻法的手続を行います。
 ```

　「最終通告」や「法的手続」等の恐怖を煽る単語を複数配置し、URLを付与しました。  

 * 正常メール例  
 ```
 件名：ログイン情報
 このたびは会員登録いただき、誠にありがとうございます。
 登録が完了しましたので、ログイン情報をご連絡いたします。
 ※ご注意（以下を必ずお読みください）
 ログイン情報-----------------------
 ログインURL：https://www.example.com
 ログインID：cysec.ml.train@gmail.com
 ログインパスワード：password'
 ```

　このメールは全体的に柔らかい単語を使用しています。  
　ちなみに、本メールは筆者がとあるサービスで会員登録した際に実際に受信したメールを基にしています（もちろんスパムメールではありません）。  

　これで、学習データとテストデータの準備が完了しました。  
　次項では実際にサンプルコードを実行し、実際にスパムメールと正常メールを正しく判別できるのか検証します。  

### 2.2. サンプルコード及び実行結果
#### 2.2.1. サンプルコード
　本ブログではPython3を使用し、簡易的なスパム検知システムを実装しました。  
　本システムの大まかな処理フローは以下のとおりです。  

 1. 学習データのロード  
 2. ナイーブベイズモデルの作成  
 3. 受診メールの取得  
 4. 受信メールのスパム判定  

```
#!/bin/env python
# -*- coding: utf-8 -*-
import codecs
import urllib.request
import nltk
import mailbox
import numpy as np
import pandas as pd
from email.header import decode_header
from nltk.corpus.reader import *
from nltk.corpus.reader.util import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Max target mail number.
MAX_COUNT = 5


# Get stopwords (Japanese and symbols).
def get_stopwords():
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(url)
    slothlib_stopwords = [line.decode('utf-8').strip() for line in slothlib_file]
    jpn_stopwords = [ss for ss in slothlib_stopwords if not ss == u'']
    jpn_symbols = ['’', '”', '‘', '。', '、', 'ー', '！', '？', '：', '；', '（', '）', '＊', '￥']
    eng_symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']
    return jpn_stopwords + jpn_symbols + eng_symbols


# Judge Spam.
def judge_spam(X_train, y_train, stop_words, message, jp_sent_tokenizer, jp_chartype_tokenizer):
    fout = codecs.open('..\\dataset\\temp.txt', 'w', 'utf-8')
    fout.write(message)
    fout.close()
    spam = PlaintextCorpusReader('..\\dataset\\', r'temp.txt', encoding='utf-8',
                                 para_block_reader=read_line_block,
                                 sent_tokenizer=jp_sent_tokenizer,
                                 word_tokenizer=jp_chartype_tokenizer)

    # Tokenize.
    tokenize_message = '|'.join(spam.words())
    lst_bow = tokenize_message.replace('\n', '').replace('\r', '').split('|')
    while lst_bow.count('') > 0:
        lst_bow.remove('')

    # Remove stop words.
    vectorizer = CountVectorizer(stop_words=stop_words)

    # Fit (Train).
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)

    # Test.
    X_test = vectorizer.transform(lst_bow)
    y_preds = clf.predict_proba(X_test)

    # Classify using Naive Bayes (MultinomialNB).
    result = ''
    if np.sum(y_preds[:, 0]) > np.sum(y_preds[:, 1]):
        result = clf.classes_[0]
    else:
        result = clf.classes_[1]

    print('[Judgement]\nThis mail is <{}>.\n{}: {}%, {}: {}%\n'.format(result,
                                                                       clf.classes_[0],
                                                                       str(round(np.mean(y_preds[:, 0])*100, 2)),
                                                                       clf.classes_[1],
                                                                       str(round(np.mean(y_preds[:, 1])*100, 2))))


if __name__ == '__main__':
    # Load train data.
    df_data = pd.read_csv('..\\dataset\\train_spam.csv', header=None)
    X = [i[0] for i in df_data.iloc[:, [0]].values.tolist()]
    y = [i[0] for i in df_data.iloc[:, [1]].values.tolist()]

    # Setting NLTK
    jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^　「」！？。]*[！？。]')
    jp_chartype_tokenizer = nltk.RegexpTokenizer(u'([ぁ-んー]'
                                                 u'+|[ァ-ンー]'
                                                 u'+|[\u4e00-\u9FFF]'
                                                 u'+|[^ぁ-んァ-ンー\u4e00-\u9FFF]+)')

    # Get stop words.
    stop_words = get_stopwords()

    # Get inbox data (ex. Mozilla Thunderbird).
    mail_box = mailbox.mbox('Your mailbox path')
    for idx, key in enumerate(mail_box.keys()[::-1]):
        if idx + 1 > MAX_COUNT:
            break

        # Get message data and receive date.
        msg_item = mail_box.get(key)
        recv_date = msg_item.get_from()

        # Get subject from message data.
        subject = ''
        for msg_subject, enc in decode_header(msg_item['Subject']):
            if enc is None:
                subject += msg_subject
            else:
                subject += msg_subject.decode(enc, 'ignore')

        # Get message body from message data.
        content = ''
        for msg_body in msg_item.walk():
            if 'text' not in msg_body.get_content_type():
                continue
            if msg_body.get_content_charset():
                content = msg_body.get_payload(decode=True).decode(msg_body.get_content_charset(), 'ignore')
            else:
                if 'charset=shift_jis' in str(msg_body.get_payload(decode=True)):
                    content = msg_body.get_payload(decode=True).decode('cp932', 'ignore')
                else:
                    print('Cannot decode : message key={}.'.format(key))
                continue

        # Judge Spam.
        if subject != '' or content != '':
            print('-'*50)
            print('[Received date]\n{}\n\n[Message]\n件名: {}\n{}'.format(recv_date, subject, content))
            judge_spam(X, y, stop_words, subject + '。' + content, jp_sent_tokenizer, jp_chartype_tokenizer)
        else:
            print('This message is empty : message key={}.'.format(key))
```

　本ブログでは、Thunderbirdのような電子メールクライアントのメールボックスから最新5件（`MAX_COUNT`）のメールを取得してスパム判定します。  

#### 2.2.2. コード解説
　今回はナイーブベイズの実装に、機械学習ライブラリの**scikit-learn**を使用しました。  
　※scikit-learnの使用方法は[公式ドキュメント](http://scikit-learn.org/)を参照のこと。  

##### パッケージのインポート
```
import nltk
...snip...
from nltk.corpus.reader import *
from nltk.corpus.reader.util import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
```

　scikit-learnのナイーブベイズ用パッケージ「`MultinomialNB`」をインポートします。  
　このパッケージには、ナイーブベイズでテキスト分類を行うための様々なクラスが収録されています。  

　また、単語分割や**ストップワード**の除去を行うために、自然言語処理用のパッケージ「`nltk`」も併せてインポートします。  

| ストップワード（Stop Word）|
|:--------------------------|
| 自然言語処理の精度向上のため、処理対象から除外せざるを得ない単語。日本語では「ここ」「は」「はい」「これら」など、英語では「is」「a」「this」「the」等のありふれた、**テキストの特徴付けに貢献しない単語や記号**等。|

##### 学習データのロード
```
# Load train data.
df_data = pd.read_csv('..\\dataset\\train_spam.csv', header=None)
X = [i[0] for i in df_data.iloc[:, [0]].values.tolist()]
y = [i[0] for i in df_data.iloc[:, [1]].values.tolist()]
```

　学習データ「[train_spam.csv](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/dataset/train_spam.csv)」から、各特徴量とラベルを取得します（特徴量：`X`、ラベル：`y`）。  

##### 単語分割の事前準備
```
# Setting NLTK
jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^　「」！？。]*[！？。]')
jp_chartype_tokenizer = nltk.RegexpTokenizer(u'([ぁ-んー]'
                                             u'+|[ァ-ンー]'
                                             u'+|[\u4e00-\u9FFF]'
                                             u'+|[^ぁ-んァ-ンー\u4e00-\u9FFF]+)')
```

　「。」「！」「？」で終わる文字列を1つの文として認識して分割するために、`nltk`の正規表現パターンを定義します（`jp_sent_tokenizer`）。  
　また、文字種を検出するために、unicodeの範囲（平仮名は [ぁ-ん] 、カタカナは [ァ-ン] 、漢字は[U+4E00 ～ U+9FFF]）を定義します（jp_chartype_tokenizer）。  

##### ストップワードの定義
```
# Get stopwords (Japanese and symbols).
def get_stopwords():
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(url)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    jpn_stopwords = [ss for ss in slothlib_stopwords if not ss == u'']
    jpn_symbols = ['’', '”', '‘', '。', '、', 'ー', '！', '？', '：', '；', '（', '）', '＊', '￥']
    eng_symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']
    return jpn_stopwords + jpn_symbols + eng_symbols

...snip...

# Get stop words.
stop_words = get_stopwords()
```

　ストップワードを定義します。  
　今回のストップワード群は、日本語ストップワードと記号（全角/半角）を組み合わせて作成します。  
　日本語のストップワードは、一般に公開されている[SlothLib](http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt)を使用します（`jpn_stopwords`）。また、記号のストップワードは、筆者がスパム検知には貢献しないと考えた幾つかの記号を定義します（`jpn_symbols`、`eng_symbols`）。  

　これらのワードや記号をリスト化して、ストップワードリストを作成します。  

##### ストップワードの削除
```
# Remove stop words.
vectorizer = CountVectorizer(stop_words=stop_words)
```

　単語をベクトル化する際に使用する`CountVectorizer`の引数'stop_words='」にストップワードリストを指定することで、単語をベクトル化するタイミングで学習データやテストデータ（受信メール）に含まれるストップワードを除去することができます。  

##### モデルの作成
```
# Fit (Train).
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
clf = MultinomialNB(alpha=1.0)
```

　学習データに含まれる単語をベクトル化します（`vectorizer.fit(X_train)`、`vectorizer.transform(X_train)`）。
　その後、`MultinomialNB`でナイーブベイズモデルを作成します。  
　`MultinomialNB`の引数として渡している「`alpha=1.0`」は、**ラプラススムージング**で加算する値を指定します。  

| ラプラススムージング（Laplace Smoothing）|
|:--------------------------|
| ナイーブベイズのような確率的言語モデルにおいて、**ゼロ頻度問題**を回避するための手法。各カテゴリの単語出現率を計算する際、テストデータに含まれる単語が各カテゴリに存在しない場合、単語の出現回数に1を加えることで、単語出現率が0になるのを防ぐ。|

| ゼロ頻度問題（Zero Frequency Problem）|
|:--------------------------|
| テストデータに含まれる単語が各カテゴリに存在しない場合、単語出現率が0になる問題。1つでも単語が含まれないと単語出現率が0になるため、正しくテキスト分類を行うことができなくなる。|

##### 学習の実行
```
clf.fit(X_train, y_train)
```

　作成したモデル`clf`の`fit`の引数として、特徴量「`X_train`」とラベル「`y_train`」を渡すことで、モデルの学習が行われます。  

##### 分類結果の取得
```
# Test.
X_test = vectorizer.transform(lst_bow)
y_preds = clf.predict_proba(X_test)
```

　受信したテストデータに含まれる単語をベクトル化します（`X_test = vectorizer.transform(lst_bow)`）。  
　ここで、引数の`lst_bow`は、受信したメール本文を単語分割し、ストップワードを除去した単語のリストです。ちなみに、ナイーブベイズは単語の出現確率などを基にテキスト分類を行う確率的言語モデルであるため、メール本文中の文法や単語の順番は考慮しません（**Bag-of-words**）。  

| Bag-of-wordsモデル（BoW）|
|:--------------------------|
| 自然言語処理で使用される文書の単純化表現。文書の構文（文法や単語の順番）は考慮せず、どのような単語が含まれているかのみを考慮する。1つの袋に様々な単語がごちゃ混ぜに入っているイメージ。|

　学習済みモデル`clf`の`predict_proba`の引数として、ベクトル化したテストデータ「`X_test`」を渡すことで、テストデータに含まれる各単語の分類確率を得る事ができます。  

##### 分類結果の出力
```
result = ''
if np.sum(y_preds[:, 0]) > np.sum(y_preds[:, 1]):
    result = clf.classes_[0]
else:
    result = clf.classes_[1]
```

　リストとして保存されている各単語の分類確率「`y_preds`」から、各カテゴリ（スパム or 正常）のスコア（各単語の分類確率の和）を比較し、スパムメールまたは正常メールであるのかを判定します。  

##### 分類確率の出力
```
print('[Judgement]\nThis mail is <{}>.\n{}: {}%, {}: {}%\n'.format(result,
                                                                   clf.classes_[0],
                                                                   str(round(np.mean(y_preds[:, 0])*100, 2)),
                                                                   clf.classes_[1],
                                                                   str(round(np.mean(y_preds[:, 1])*100, 2))))
```

　リストとして保存されている各単語の分類確率「`y_preds`」から、各カテゴリ（スパム or 正常）の分類確率（各単語の分類確率の平均）と判定結果をコンソールに出力します。  

##### モデルの評価
　上記で作成したテストデータ（スパムメールまたは正常メール）を実際に送信し、スパム判定ができるのか確認します。  

　1通目のメールは以下のとおりです。  
　これは**スパムメール**を想定して用意したテストデータです。  

　 ![スパムメール](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/img/spam_mail.png)

　これに対するスパム検知システムの出力結果は以下のとおりです。  

 ```
 [Received date]
 - Thu Sep 27 15:14:18 2018

 [Message]
 件名: 警告
 これは最終通告です。
 5分以内に以下のサイトにアクセスして、退会手続を行ってください。
 http://www.example.com

 退会しない場合は、即刻法的手続を行います。

 [Judgement]
 This mail is <Spam>.
 Normal: 46.85%, Spam: 53.15%
 ```

　分類確率（`Probabilities`）を見ると、カテゴリ「`Normal`」は**46.85%**、カテゴリ「`Spam`」は**53.15**%であることが分かります。よって、カテゴリ「`Spam`」の確率が上回っているため、1通目のメールは**スパムメール**であると判定されます（`This text is <Spam>`）。  

　2通目のメールは以下のとおりです。  
　これは**正常メール**を想定して用意したテストデータです。  

　![正常メール](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/img/normal_mail.png)

　これに対するスパム検知システムの出力結果は以下のとおりです。  

 ```
 [Received date]
 - Thu Sep 27 15:16:12 2018

 [Message]
 件名: ログイン情報
 このたびは会員登録いただき、誠にありがとうございます。
 登録が完了しましたので、ログイン情報をご連絡いたします。
 ※ご注意（以下を必ずお読みください）
 ログイン情報-----------------------
 ログインURL：https://www.example.com
 ログインID：cysec.ml.train@gmail.com
 ログインパスワード：password'

 [Judgement]
 This mail is <Normal>.
 Normal: 50.4%, Spam: 49.6%
 ```

　分類確率（`Probabilities`）を見ると、カテゴリ「`Normal`」は**50.4%**、カテゴリ「`Spam`」は**49.6**%であることが分かります。よって、カテゴリ「`Normal`」の確率が上回っているため、2通目のメールは**正常メール**であると判定されます（`This text is <Normal>`）。  

　この結果から、簡易的な実装であるものの、受信したメールを正しくスパム判定できることが分かりました。  
　本ブログではナイーブベイズをスパム判定に利用しましたが、他のテキスト分類を利用するタスクにも利用可能です。例えば、筆者はWebクローラーにおける**ページ種別判定エンジン**にナイーブベイズを利用したことがあります（詳細は[こちら](https://www.mbsd.jp/blog/20160113.html)）。また、とあるブログサイトでは、利用者が投稿したブログのカテゴリー分けに利用されています。  

　このようにナイーブベイズの利用用途は広いため、ご自身で様々なタスクに利用してみる事をお勧めします。  

#### 2.3. 動作条件
 * Python 3.6.1（Anaconda3）
 * nltk 3.3.0
 * numpy 1.15.1
 * scikit-learn 0.20.0
