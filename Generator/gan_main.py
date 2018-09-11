# -*- coding: utf-8 -*-
import os
import sys
import random
import codecs
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
    def __init__(self, template, browser):
        self.util = Utilty()
        self.template = template
        self.obj_browser = browser

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
        self.html_file = config['Common']['gan_html_file']
        self.result_dir = self.util.join_path(full_path, config['Common']['result_dir'])

        # Genetic Algorithm setting value.
        self.genom_length = int(config['Genetic']['genom_length'])
        self.gene_dir = self.util.join_path(full_path, config['Genetic']['gene_dir'])
        self.genes_path = self.util.join_path(self.gene_dir, config['Genetic']['gene_file'])
        self.ga_result_file = config['Genetic']['result_file']
        self.eval_place_list = config['Genetic']['html_eval_place'].split('@')

        # Generative Adversarial Network setting value.
        self.input_size = int(config['GAN']['input_size'])
        self.batch_size = int(config['GAN']['batch_size'])
        self.num_epoch = int(config['GAN']['num_epoch'])
        self.max_sig_num = int(config['GAN']['max_sig_num'])
        self.max_synthetic_num = int(config['GAN']['max_synthetic_num'])
        self.weight_dir = self.util.join_path(full_path, config['GAN']['weight_dir'])
        self.gen_weight_file = config['GAN']['generator_weight_file']
        self.dis_weight_file = config['GAN']['discriminator_weight_file']
        self.gan_result_file = config['GAN']['result_file']
        self.gan_vec_result_file = config['GAN']['vec_result_file']
        self.generator = None
        self.flt_size = 0

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
    def train(self, df_genes, list_sigs):
        X_train = []
        self.flt_size = len(df_genes)/2.0

        # Load train data (=ga result).
        X_train = np.array(list_sigs)
        X_train = (X_train.astype(np.float32) - self.flt_size)/self.flt_size

        # Build discriminator.
        discriminator = self.discriminator_model()
        d_opt = SGD(lr=0.1, momentum=0.1, decay=1e-5)
        discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

        # Build generator and discriminator (fixed weight of discriminator).
        discriminator.trainable = False
        self.generator = self.generator_model()
        dcgan = Sequential([self.generator, discriminator])
        g_opt = SGD(lr=0.1, momentum=0.3)
        dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

        # Execute train.
        num_batches = int(len(X_train) / self.batch_size)
        lst_scripts = []
        for epoch in range(self.num_epoch):
            for batch in range(num_batches):
                # Create noise for inputting to generator.
                noise = np.array([np.random.uniform(-1, 1, self.input_size) for _ in range(self.batch_size)])

                # Generate new injection code using noise.
                generated_codes = self.generator.predict(noise, verbose=0)

                # Update weight of discriminator.
                image_batch = X_train[batch * self.batch_size:(batch + 1) * self.batch_size]
                X = image_batch
                y = [random.uniform(0.7, 1.2) for _ in range(self.batch_size)]
                d_loss = discriminator.train_on_batch(X, y)
                X = generated_codes
                y = [random.uniform(0.0, 0.3) for _ in range(self.batch_size)]
                d_loss = discriminator.train_on_batch(X, y)

                # Update weight of generator.
                noise = np.array([np.random.uniform(-1, 1, self.input_size) for _ in range(self.batch_size)])
                g_loss = dcgan.train_on_batch(noise, [1]*self.batch_size)

                # Build HTML syntax from generated codes.
                for generated_code in generated_codes:
                    str_html = ''
                    lst_genom = []
                    for gene_num in generated_code:
                        gene_num = (gene_num * self.flt_size) + self.flt_size
                        gene_num = int(np.round(gene_num))
                        if gene_num == len(df_genes):
                            gene_num -= 1
                        lst_genom.append(int(gene_num))
                    str_html = self.util.transform_gene_num2str(df_genes, lst_genom)
                    self.util.print_message(OK, 'Train GAN : epoch={}, batch={}, g_loss={}, d_loss={}, {} ({})'.
                                            format(epoch, batch, g_loss, d_loss,
                                                   np.round((generated_code * self.flt_size) + self.flt_size),
                                                   str_html))

                    # Save gene information when last epoch.
                    if epoch == (self.num_epoch - 1):
                        lst_scripts.append([lst_genom, str_html, noise])

            # Save weights of network each epoch.
            self.generator.save_weights(self.util.join_path(self.weight_dir,
                                                            self.gen_weight_file.replace('*', str(epoch))))
            discriminator.save_weights(self.util.join_path(self.weight_dir,
                                                           self.dis_weight_file.replace('*', str(epoch))))

        return lst_scripts

    # Mean of two vectors.
    def vector_mean(self, vector1, vector2):
        return (vector1 + vector2)/2

    # Main control.
    def main(self):
        # Load created individuals by Genetic Algorithm.
        sig_path = self.util.join_path(self.result_dir, self.ga_result_file.replace('*', self.obj_browser.name))
        df_temp = pd.read_csv(sig_path, encoding='utf-8').fillna('')
        df_sigs = df_temp[~df_temp.duplicated()]

        if len(df_sigs) != 0:
            list_sigs = []
            # Extract genom list from ga result.
            for idx in range(len(df_sigs)):
                list_temp = df_sigs['sig_vector'].values[idx].replace('[', '').replace(']', '').split(',')
                list_sigs.append([int(s) for s in list_temp])

            # Load gene list.
            df_genes = pd.read_csv(self.genes_path, encoding='utf-8').fillna('')

            # Generate individuals (=injection codes).
            lst_scripts = []
            for target_sig in list_sigs:
                self.util.print_message(NOTE, 'Start generating injection codes using {}'.format(target_sig))
                target_sig_list = [target_sig for _ in range(self.max_sig_num)]
                lst_scripts.extend(self.train(df_genes, target_sig_list))

            # Create saving file (only header).
            save_path = self.util.join_path(self.result_dir, self.gan_result_file.replace('*', self.obj_browser.name))
            if os.path.exists(save_path) is False:
                pd.DataFrame([], columns=['eval_place', 'sig_vector', 'sig_string']).to_csv(save_path,
                                                                                            mode='w',
                                                                                            header=True,
                                                                                            index=False)

            # Evaluate generated individual.
            result_list = []
            valid_noise_list = []
            for idx, indivisual in enumerate(lst_scripts):
                self.util.print_message(NOTE, '{}/{} Evaluate individual : {} ({})'.format(idx + 1,
                                                                                           len(lst_scripts),
                                                                                           indivisual[0],
                                                                                           indivisual[1]))
                for eval_place in self.eval_place_list:
                    # Build html syntax.
                    html = self.template.render({eval_place: indivisual[1]})
                    eval_html_path = self.util.join_path(self.html_dir, self.html_file)
                    with codecs.open(eval_html_path, 'w', encoding='utf-8') as fout:
                        fout.write(html)

                    # Evaluate individual using selenium.
                    selenium_score, error_flag = self.util.check_individual_selenium(self.obj_browser,
                                                                                     eval_html_path)
                    if error_flag:
                        continue

                    # Check generated individual using selenium.
                    if selenium_score > 0:
                        self.util.print_message(WARNING, 'Detect running script: "{}" in {}.'.format(indivisual[1],
                                                                                                     eval_place))

                        # Save running script.
                        result_list.append([eval_place, indivisual[0], indivisual[1]])
                        valid_noise_list.append(indivisual[2])
                    else:
                        self.util.print_message(FAIL, 'Not detect running script: "{}" in {}.'.format(indivisual[1],
                                                                                                      eval_place))

                # Save generated individuals.
                pd.DataFrame(result_list).to_csv(save_path, mode='a', header=False, index=False)

            # Calculate of tensor mean.
            if len(valid_noise_list) != 0:
                vector_result_list = []
                for idx in range(self.max_synthetic_num):
                    self.util.print_message(NOTE, 'Start synthesizing individuals.')
                    vector_idx1 = np.random.randint(0, len(valid_noise_list) - 1)
                    vector_idx2 = np.random.randint(0, len(valid_noise_list) - 1)
                    self.util.print_message(OK, '{}/{} Synthetic of two individuals : ({}) + ({}).'.
                                            format(idx + 1,
                                                   self.max_synthetic_num,
                                                   [vector_idx1][2],
                                                   result_list[vector_idx2][2]))
                    mean_noise = self.vector_mean(valid_noise_list[vector_idx1], valid_noise_list[vector_idx2])
                    generated_codes = self.generator.predict(mean_noise, verbose=0)

                    # Build HTML syntax from generated codes.
                    for generated_code in generated_codes:
                        str_html = ''
                        lst_genom = []
                        for gene_num in generated_code:
                            gene_num = (gene_num * self.flt_size) + self.flt_size
                            gene_num = int(np.round(gene_num))
                            if gene_num == len(df_genes):
                                gene_num -= 1
                            lst_genom.append(int(gene_num))
                        str_html = self.util.transform_gene_num2str(df_genes, lst_genom)

                        for eval_place in self.eval_place_list:
                            # Build html syntax.
                            html = self.template.render({eval_place: str_html})
                            eval_html_path = self.util.join_path(self.html_dir, self.html_file)
                            with codecs.open(eval_html_path, 'w', encoding='utf-8') as fout:
                                fout.write(html)

                            # Evaluate individual using selenium.
                            selenium_score, error_flag = self.util.check_individual_selenium(self.obj_browser,
                                                                                             eval_html_path)
                            if error_flag:
                                continue

                            # Check synthetic individual using selenium.
                            if selenium_score > 0:
                                self.util.print_message(WARNING,
                                                        'Detect running script: "{}" in {}.'.format(str_html,
                                                                                                    eval_place))

                                # Save running script.
                                vector_result_list.append([str_html,
                                                           result_list[vector_idx1][1],
                                                           result_list[vector_idx2][1]])
                            else:
                                self.util.print_message(FAIL, 'Not detect running script: "{}" in {}.'.
                                                        format(str_html, eval_place))

                # Save synthesized individuals.
                save_path = self.util.join_path(self.result_dir, self.gan_vec_result_file.
                                                replace('*', self.obj_browser.name))
                pd.DataFrame(vector_result_list,
                             columns=['synthesized_script', 'origin_script1', 'origin_script2']).to_csv(save_path,
                                                                                                        mode='w',
                                                                                                        header=True,
                                                                                                        index=False)

            self.util.print_message(NOTE, 'Done generation of injection codes using Generative Adversarial Networks.')
        else:
            self.util.print_message(WARNING, 'Signature of {} do not include.'.format(sig_path))
            self.util.print_message(WARNING, 'Skip process of Generative Adversarial Networks')
