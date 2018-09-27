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
