#!/usr/bin/python
# coding:utf-8
import os
import configparser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils


class Recommend:
    def __init__(self, state_size=17, save_name='recommender'):
        # Read config.ini.
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(full_path, 'config.ini'))
        except FileExistsError as err:
            print('[-] File exists error: {}', err)
            exit(1)
        self.epoch = int(config['Recommender']['epoch'])
        self.batch_size = int(config['Recommender']['batch_size'])
        self.trained_path = os.path.join(full_path, config['Recommender']['trained_path'])
        if os.path.exists(self.trained_path) is False:
            os.mkdir(self.trained_path)
        self.xss_train_data = os.path.join(full_path, config['Recommender']['xss_train_data'])
        self.state_size = state_size
        self.save_path = os.path.join(self.trained_path, save_name)

        # Load train data, create label.
        self.X_train, self.Y_train, self.colums, self.classes, self.y_data, self.output_size = self.load_train_data()

    # Build Multilayer Perceptron.
    def build_multilayer_perceptron(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.output_size))
        model.add(Activation('softmax'))
        return model

    # Load training data and create label.
    def load_train_data(self):
        X_train = []
        Y_train = []
        tmp_classes = []

        # Load training data.
        print('[*] Loading train data and Create label.')
        obj_train = pd.read_csv(self.xss_train_data, encoding='utf-8').fillna('')
        df_X_train = obj_train.iloc[:, 0: 17]
        for idx in range(len(df_X_train)):
            X_train.append(list(df_X_train.loc[idx].values.flatten()))
        df_Y_train = obj_train.iloc[:, [17, 18]]
        for idx in range(len(df_Y_train)):
            Y_train.append(list(df_Y_train.iloc[idx, [0]].values.flatten()))
            tmp_classes.append(df_Y_train.iloc[idx, [0]].values.flatten()[0])

        # Create label.
        classes = list(set(tmp_classes))

        return X_train, Y_train, len(df_X_train.columns), classes, df_Y_train, len(classes)

    # Train model.
    def training_model(self):
        print('[*] Training model..')
        batch_size = self.batch_size
        nb_epoch = self.epoch

        # Convert training data to 1-dimensional array.
        X_train = np.array(self.X_train, np.ndarray)
        X_train = X_train.reshape(len(X_train), self.colums)
        print('[+] Train samples: {}'.format(X_train.shape[0]))

        # One-hot-encoding.
        Y_train = np_utils.to_categorical(self.Y_train, self.output_size)

        # Divide to training data and testing data.
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.8)

        # Build Multilayer Perceptron (MLP).
        model = self.build_multilayer_perceptron()

        # Display summary of MLP structure.
        model.summary()

        # Compile model.
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        # Execute train.
        _ = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      verbose=1,
                      validation_split=0.1)
        print('[*] Finish training.')

        # Evaluate.
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
        print('[*] Evaluation.')
        print('[+] Test loss:', loss)
        print('[+] Test acc:', accuracy)

        # Save trained model(.h5).
        model.save_weights('{}.h5'.format(self.save_path), True)
        print('[+] Saved model: {}.h5'.format(self.save_path))

    def predict(self, lst_feature):
        # Build Multilayer Perceptron (MLP).
        model = self.build_multilayer_perceptron()

        # Load trained model(.h5).
        model.load_weights('{}.h5'.format(self.save_path))
        print('[*] Loading trained data: {}.h5'.format(self.save_path))

        # Compile model.
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        # Convert testing data to 1-dimensional array.
        X_test = np.array(lst_feature, np.ndarray)
        X_test = X_test.reshape(1, 17)

        # Predict.
        pred = model.predict(X_test)[0]
        top = 3
        top_indices = pred.argsort()[-top:][::-1]
        result = [(self.classes[i], pred[i]) for i in top_indices]

        # Display prediction result.
        print('[*] Recommended.')
        print('[+] Rank\tInjection code\tProbability')
        print('='*80)
        for idx, item in enumerate(result):
            injection_code = self.y_data[self.y_data['label'] == (int(item[0]))].values.flatten()[1]
            injection_code = injection_code[1:-1].replace('\\n', '%0a').replace('\\r', '%0d')
            print('[+] {}\t{}\t{}'.format(idx+1, injection_code, item[1]))
