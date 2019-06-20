#!/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Parameters of LDA
n_samples = 100000
n_features = 100
n_topics = 10
n_top_words = 10
max_df = 0.95
min_df = 10


# Output topic and topic words
def print_top_words(model, feature_names):
    for topic_idx, topic in enumerate(model.components_):
        message = 'Topic #{0}: '.format(topic_idx)
        message += ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)


if __name__ == '__main__':
    full_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(full_path, '..')

    # Get original stop words
    dataset_path = os.path.join(os.path.join(root_path, 'dataset'), 'topic_model')
    stop_word_path = os.path.join(dataset_path, 'stop_words.txt')
    df_stop_words = pd.read_csv(stop_word_path, encoding='utf-8').fillna('')
    data_samples = df_stop_words.loc[:].values
    lst_stop_words = [word[0] for word in data_samples.tolist()]

    # Load cve description
    lst_cve_description = []
    for file in os.listdir(dataset_path):
        # extract description from 'cve****.txt'
        if file[:3] == 'cve' and file[-4:] == '.csv':
            df_cve = pd.read_csv(os.path.join(dataset_path, file), encoding='utf-8').fillna('')
            data_samples = df_cve.loc[random.sample(range(len(df_cve)), n_samples)].values
            lst_cve_description = [description[0] for description in data_samples.tolist()]
        else:
            continue

        # Remove stop words
        lst_bow = []
        for description in lst_cve_description:
            temp_description = ''
            for word in description.lower().split(' '):
                if not word in lst_stop_words:
                    temp_description += word + ' '
            lst_bow.append(temp_description)

        # Use tf (raw term count) features for LDA (Latent Dirichlet Allocation).
        tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,
                                        max_features=n_features,
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform(lst_bow)
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)

        # Output result
        print('#'*50)
        print('Topics in LDA model from ' + file)
        print('#'*50)
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names)
        print('')

    print('finish!!')
