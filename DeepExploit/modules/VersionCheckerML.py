#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import codecs
import configparser
import pickle
from .NaiveBayes import NaiveBayes

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


class VersionCheckerML:
    def __init__(self, utility):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.join(self.full_path, '../')
        try:
            config.read(os.path.join(self.root_path, 'config.ini'))
        except Exception as e:
            self.utility.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

        self.category_type = config['VersionCheckerML']['category']
        self.train_path = os.path.join(self.full_path, config['VersionCheckerML']['train_path'])
        self.trained_path = os.path.join(self.full_path, config['VersionCheckerML']['trained_path'])
        self.train_os_in = os.path.join(self.train_path, config['VersionCheckerML']['train_os_in'])
        self.train_os_out = os.path.join(self.trained_path, config['VersionCheckerML']['train_os_out'])
        self.train_web_in = os.path.join(self.train_path, config['VersionCheckerML']['train_web_in'])
        self.train_web_out = os.path.join(self.trained_path, config['VersionCheckerML']['train_web_out'])
        self.train_framework_in = os.path.join(self.train_path, config['VersionCheckerML']['train_framework_in'])
        self.train_framework_out = os.path.join(self.trained_path, config['VersionCheckerML']['train_framework_out'])
        self.train_cms_in = os.path.join(self.train_path, config['VersionCheckerML']['train_cms_in'])
        self.train_cms_out = os.path.join(self.trained_path, config['VersionCheckerML']['train_cms_out'])
        return

        # Identify product name using ML.
    def identify_product(self, parsed, response, client):
        product_list = []

        try:
            # Predict product name each category (OS, Middleware, CMS..).
            list_category = self.category_type.split('@')
            for category in list_category:
                # Keep Alive to Metasploit.
                client.keep_alive()

                # Learning.
                nb = None
                if category == 'OS':
                    nb = self.train(self.train_os_in, self.train_os_out)
                elif category == 'WEB':
                    nb = self.train(self.train_web_in, self.train_web_out)
                elif category == 'FRAMEWORK':
                    nb = self.train(self.train_framework_in, self.train_framework_out)
                elif category == 'CMS':
                    nb = self.train(self.train_cms_in, self.train_cms_out)
                else:
                    self.utility.print_message(FAIL, 'Choose category is not found.')
                    exit(1)

                # Predict product name.
                product, prob, keyword_list, classified_list = nb.classify(response)

                # Output result of prediction (body).
                # If no feature, result is unknown.
                if len(keyword_list) != 0:
                    # Add product information.
                    port_num = 80
                    path = os.path.split(parsed.path)[0]
                    if parsed.port is not None:
                        port_num = parsed.port
                    if path.endswith('/') is False:
                        path += '/'
                    product_list.append(product + '@*@' + str(port_num) + '@' + path)
                    msg = 'Predict product={}/{}%, verson={}, trigger={}'.format(product, prob, '*', keyword_list)
                    self.utility.print_message(OK, msg)
                    self.utility.print_message(NOTE, 'category : {}'.format(category))
        except Exception as e:
            msg = 'Identifying product is failure : {}'.format(e)
            self.utility.print_exception(e, msg)

        return product_list

    # Classifier product name using Machine Learning.
    def get_product_name(self, parsed, response, client):
        self.utility.print_message(NOTE, 'Analyzing gathered HTTP response using ML.')

        # Execute classifier.
        product_list = self.identify_product(parsed, response, client)
        if len(product_list) == 0:
            self.utility.print_message(WARNING, 'Product Not Found.')

        return product_list

    # Execute learning / Get learned data.
    def train(self, in_file, out_file):
        # If existing learned data (pkl), load learned data.
        nb = None
        if os.path.exists(out_file):
            with open(out_file, 'rb') as f:
                nb = pickle.load(f)
        # If no learned data, execute learning.
        else:
            # Read learning data.
            nb = NaiveBayes()
            with codecs.open(in_file, 'r', 'utf-8') as fin:
                lines = fin.readlines()
                items = []

                for line in lines:
                    words = line[:-2]
                    train_words = words.split('@')
                    items.append(train_words[1])
                    nb.train(train_words[3], train_words[0])

            # Save learned data to pkl file.
            with open(out_file, 'wb') as f:
                pickle.dump(nb, f)
        return nb
