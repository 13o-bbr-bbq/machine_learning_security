# -*- coding: utf-8 -*-
import os
import sys
import random
import csv
import configparser
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout
from keras import backend as K
from util import Utilty

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

# Setting of GAN.
K.set_image_dim_ordering('th')


# Generative Adversarial Networks.
class GAN:
    def __init__(self):
        self.util = Utilty()

        # Read config.ini.
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(self.util.join_path(full_path, 'config.ini'))
        except FileExistsError as e:
            self.util.print_message(FAIL, 'File exists error: {}'.format(e))
            sys.exit(1)

        # Common setting value.
        self.wait_time = float(config['Common']['wait_time'])
        self.html_dir = self.util.join_path(full_path, config['Common']['html_dir'])
        self.html_file = config['Common']['html_file']
        self.result_dir = self.util.join_path(full_path, config['Common']['result_dir'])

        # Genetic Algorithm setting value.
        self.genom_length = int(config['Genetic']['genom_length'])
        self.gene_dir = self.util.join_path(full_path, config['Genetic']['gene_dir'])
        self.genes_path = self.util.join_path(self.gene_dir, config['Genetic']['gene_file'])
        self.result_file = config['Genetic']['result_file']

        # Generative Adversarial Network setting value.
        self.input_size = int(config['GAN']['input_size'])
        self.batch_size = int(config['GAN']['batch_size'])
        self.num_epoch = int(config['GAN']['num_epoch'])

    # Build generator model.
    def generator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=self.input_size, output_dim=self.input_size*10, init='glorot_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(self.input_size*10, init='glorot_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(self.input_size*5, init='glorot_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=self.genom_length, init='glorot_uniform'))
        model.add(Activation('tanh'))
        return model

    # Build discriminator model.
    def discriminator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=self.genom_length, output_dim=self.genom_length*10, init='glorot_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dense(self.genom_length*10, init='glorot_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1, init='glorot_uniform'))
        model.add(Activation('sigmoid'))
        return model

    # Train GAN model (generate injection codes).
    def train(self, df_genes, df_sigs):
        X_train = []
        flt_size = len(df_genes)/2.0

        # load your train data
        X_train.append(df_sigs[1])
        X_train = np.array(X_train)
        X_train = (X_train.astype(np.float32) - flt_size)/flt_size

        discriminator = self.discriminator_model()
        d_opt = SGD(lr=0.1, momentum=0.1, decay=1e-5)
        discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

        # Generator + Discriminator (fixed weight of discriminator).
        discriminator.trainable = False
        generator = self.generator_model()
        dcgan = Sequential([generator, discriminator])
        g_opt = SGD(lr=0.1, momentum=0.3)
        dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

        num_batches = int(len(X_train) / self.batch_size)
        lst_scripts = []
        for epoch in range(self.num_epoch):
            for batch in range(num_batches):
                noise = np.array([np.random.uniform(-1, 1, self.input_size) for _ in range(self.batch_size)])
                generated_images = generator.predict(noise, verbose=0)

                # Update weight of discriminator.
                image_batch = X_train[batch * self.batch_size:(batch + 1) * self.batch_size]
                X = image_batch
                y = [random.uniform(0.7, 1.2) for _ in range(self.batch_size)]
                d_loss = discriminator.train_on_batch(X, y)
                X = generated_images
                y = [random.uniform(0.0, 0.3) for _ in range(self.batch_size)]
                d_loss = discriminator.train_on_batch(X, y)

                # Update weight of generator.
                noise = np.array([np.random.uniform(-1, 1, self.input_size) for _ in range(self.batch_size)])
                g_loss = dcgan.train_on_batch(noise, [1]*self.batch_size)
                for generated_image in generated_images:
                    str_html = ''
                    for gene_num in generated_image:
                        gene_num = (gene_num*flt_size)+flt_size
                        gene_num = np.round(gene_num)
                        if gene_num == len(df_genes):
                            gene_num -= 1
                        str_html += str(df_genes.loc[gene_num].values[0])
                    lst_scripts.append(str_html)
                    self.util.print_message(OK, '{},{},{},{},{},{}'.
                                            format(epoch,
                                                   batch,
                                                   g_loss,
                                                   d_loss,
                                                   np.round((generated_image*flt_size)+flt_size),
                                                   str_html))

            generator.save_weights('.\\weight\\generator_' + str(epoch) + '.h5')
            discriminator.save_weights('.\\weight\\discriminator_' + str(epoch) + '.h5')

        return lst_scripts

    def main(self, obj_browser, eval_place):
        sig_path = self.util.join_path(self.result_dir, self.result_file.replace('*', obj_browser.name))
        df_sigs = pd.read_csv(sig_path, encoding='utf-8').fillna('')
        df_selected_sigs = df_sigs[(df_sigs[0] == eval_place)]

        if len(df_selected_sigs) != 0:
            df_genes = pd.read_csv(self.genes_path, encoding='utf-8').fillna('')

            # Generate injection codes.
            lst_scripts = self.train(df_genes, df_sigs)
            self.util.print_message(NOTE, 'Generated injection codes: {}'.format(lst_scripts))
        else:
            self.util.print_message(WARNING, 'Signature of {} do not include in {}.'.format(eval_place, sig_path))
            self.util.print_message(WARNING, 'Skip process of Generative Adversarial Networks')
