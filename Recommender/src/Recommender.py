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

# class
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
           '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
           '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
           '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
           '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
           '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
           '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
           '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
           '121', '122', '123', '124', '125', '126', '127', '128', '129', '130',
           '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
           '141']


class Recommend:
    def __init__(self, state_size=17, save_name='recommender'):
        # Read config.ini.
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(full_path, 'config.ini'))
        except FileExistsError as err:
            print('[-] File exists error: {0}', err)
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

    def build_multilayer_perceptron(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.output_size))
        model.add(Activation('softmax'))
        return model

    def load_train_data(self):
        X_train = []
        Y_train = []
        tmp_classes = []
        tmp_codes = []

        # load training data
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
        output_size = len(classes) + 1

        return X_train, Y_train, len(df_X_train.columns), classes, df_Y_train, output_size

    def training_model(self):
        print('[*] Training model..')
        batch_size = self.batch_size
        nb_epoch = self.epoch

        # convert training data to 1-dimensional array
        X_train = np.array(self.X_train, np.ndarray)
        X_train = X_train.reshape(len(X_train), self.colums)
        print('[+] Train samples: {}'.format(X_train.shape[0]))

        # one-hot-encoding
        Y_train = np_utils.to_categorical(self.Y_train)

        # divide to training data and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.8)

        # build mlp
        model = self.build_multilayer_perceptron()

        # display sammary of mlp structure
        model.summary()

        # compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        # train
        _ = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      verbose=1,
                      validation_split=0.1)

        # evaluate
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
        print('[+] Test loss:', loss)
        print('[+] Test acc:', accuracy)

        # save trained model(.h5)
        model.save_weights('{}.h5'.format(self.save_path), True)
        print('[+] Saved model: {}.h5'.format(self.save_path))

    def predict(self, lst_feature):
        # builds mlp
        model = self.build_multilayer_perceptron()

        # load trained model(.h5)
        model.load_weights('{}.h5'.format(self.save_path))
        print('[*] Loading learned data from {}.h5'.format(self.save_path))

        # compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        # convert testing data to 1-dimensional array
        X_test = np.array(lst_feature, np.ndarray)
        X_test = X_test.reshape(1, 17)

        # predict
        pred = model.predict(X_test)[0]
        top = 3
        top_indices = pred.argsort()[-top:][::-1]
        result = [(classes[i-1], pred[i]) for i in top_indices]

        # Display prediction result.
        for idx, x in enumerate(result):
            injection_code = self.y_data[self.y_data['label'] == (int(x[0])-1)].values.flatten()[1]
            print('[+] Rank {}: {} / Probability: {}'.format(idx+1, injection_code, x[1]))
