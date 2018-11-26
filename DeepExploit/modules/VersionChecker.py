#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import codecs
import re
import configparser
from util import Utilty

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


class VersionChecker:
    def __init__(self, utility):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.join(self.full_path, '../')
        config.read(os.path.join(self.root_path, 'config.ini'))

        try:
            self.signatures_dir = os.path.join(self.root_path, config['Common']['signature_path'])
            self.signature_file = os.path.join(self.signatures_dir, config['VersionChecker']['signature_file'])
        except Exception as e:
            self.utility.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

    # Identify product name using signature.
    def identify_product(self, parsed, response, client):
        product_list = []

        try:
            # Identify product name and version.
            with codecs.open(self.signature_file, 'r', 'utf-8') as fin:
                matching_patterns = fin.readlines()
                for idx, pattern in enumerate(matching_patterns):
                    # Keep Alive to Metasploit per 10 count.
                    if (idx + 1) % 10 == 0:
                        client.keep_alive()

                    items = pattern.replace('\r', '').replace('\n', '').split('@')
                    product = items[0].lower()
                    default_ver = items[1]
                    signature = items[2]
                    self.utility.print_message(OK, '{}/{} Check {} using [{}]'.format(idx+1,
                                                                                      len(matching_patterns),
                                                                                      product,
                                                                                      signature))
                    obj_match = re.search(signature, response, flags=re.IGNORECASE)
                    if obj_match is not None:
                        # Check version.
                        version = default_ver
                        if obj_match.re.groups > 1:
                            version = obj_match.group(2)

                        # Add product information.
                        port_num = 80
                        path = os.path.split(parsed.path)[0]
                        if parsed.port is not None:
                            port_num = parsed.port
                        if path.endswith('/') is False:
                            path += '/'
                        product_list.append(product + '@' + version + '@' + str(port_num) + '@' + path)

                        # product_list.append([category, vendor, product, version, trigger])
                        msg = 'Find product={}/{}'.format(product, version)
                        self.utility.print_message(WARNING, msg)
        except Exception as e:
            msg = 'Identifying product is failure : {}'.format(e)
            self.utility.print_exception(e, msg)

        return product_list

    # Classifier product name using signatures.
    def get_product_name(self, parsed, response, client):
        self.utility.print_message(NOTE, 'Analyzing gathered HTTP response.')

        # Execute classifier.
        product_list = self.identify_product(parsed, response, client)
        if len(product_list) == 0:
            self.utility.print_message(WARNING, 'Product Not Found.')

        return product_list
