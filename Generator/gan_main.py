# -*- coding: utf-8 -*-
import sys
import random
import csv
import ConfigParser
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout
from keras import backend as K

# gan
K.set_image_dim_ordering('th')
INPUT_SIZE = 200
GEN_OUTPUT_SIZE = 5
BATCH_SIZE = 32
NUM_EPOCH = 50


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=INPUT_SIZE, output_dim=INPUT_SIZE*10, init='glorot_uniform'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(INPUT_SIZE*10, init='glorot_uniform'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(INPUT_SIZE*5, init='glorot_uniform'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=GEN_OUTPUT_SIZE, init='glorot_uniform'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Dense(input_dim=GEN_OUTPUT_SIZE, output_dim=GEN_OUTPUT_SIZE*10, init='glorot_uniform'))
    model.add(LeakyReLU(0.2))
    model.add(Dense(GEN_OUTPUT_SIZE*10, init='glorot_uniform'))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, init='glorot_uniform'))
    model.add(Activation('sigmoid'))
    return model


def train(df_genes):
    X_train = []
    flt_size = len(df_genes)/2.0
    # load your train data
    with open('.\\signature\\xss_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_train.append(row)
    X_train = np.array(X_train)
    X_train = (X_train.astype(np.float32) - flt_size)/flt_size

    discriminator = discriminator_model()
    d_opt = SGD(lr=0.1, momentum=0.1, decay=1e-5)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator+discriminator (fixed weight of discriminator)
    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = SGD(lr=0.1, momentum=0.3)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(len(X_train) / BATCH_SIZE)
    lst_scripts = []
    for epoch in range(NUM_EPOCH):
        for batch in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, INPUT_SIZE) for _ in range(BATCH_SIZE)])
            generated_images = generator.predict(noise, verbose=0)

            # update weight of discriminator
            image_batch = X_train[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
            X = image_batch
            y = [random.uniform(0.7, 1.2) for _ in range(BATCH_SIZE)]
            d_loss = discriminator.train_on_batch(X, y)
            X = generated_images
            y = [random.uniform(0.0, 0.3) for _ in range(BATCH_SIZE)]
            d_loss = discriminator.train_on_batch(X, y)

            # update weight of generator
            noise = np.array([np.random.uniform(-1, 1, INPUT_SIZE) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            for generated_image in generated_images:
                str_html = ''
                for gene_num in generated_image:
                    gene_num = (gene_num*flt_size)+flt_size
                    gene_num = np.round(gene_num)
                    if gene_num == len(df_genes):
                        gene_num -= 1
                    str_html += str(df_genes.loc[gene_num].values[0])
                lst_scripts.append(str_html)
                print('{0},{1},{2},{3},{4},{5}'.format(epoch,
                                                       batch,
                                                       g_loss,
                                                       d_loss,
                                                       np.round((generated_image*flt_size)+flt_size),
                                                       str_html))

        generator.save_weights('.\\weight\\generator_' + str(epoch) + '.h5')
        discriminator.save_weights('.\\weight\\discriminator_' + str(epoch) + '.h5')

    return lst_scripts


if __name__ == "__main__":
    # load config
    inifile = ConfigParser.SafeConfigParser()
    try:
        inifile.read('config.ini')
    except:
        print('usage: can\'t read config.ini.')
        exit(1)
    genes = inifile.get('Genetic', 'gene_list')
    signature = inifile.get('GAN', 'xss_list')
    element = inifile.get('GAN', 'elem_list')
    df_genes = pd.read_csv(genes, encoding='utf-8').fillna('')
    df_sigs = pd.read_csv(signature, encoding='utf-8').fillna('')

    # generate injection codes
    lst_scripts = train(df_genes)
    print(lst_scripts)
