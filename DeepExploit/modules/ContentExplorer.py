#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import codecs
import re
import time
import urllib3
import configparser
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


class ContentExplorer:
    def __init__(self, utility):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.join(self.full_path, '../')
        config.read(os.path.join(self.root_path, 'config.ini'))

        try:
            self.signature_dir = os.path.join(self.root_path, config['Common']['signature_path'])
            self.signature_file = config['ContentExplorer']['signature_file']
            self.delay_time = float(config['ContentExplorer']['delay_time'])
        except Exception as e:
            self.utility.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

    # Check product version.
    def check_version(self, default_ver, version_pattern, response):
        # Check version.
        version = default_ver
        if version_pattern != '*':
            obj_match = re.search(version_pattern, response, flags=re.IGNORECASE)
            if obj_match is not None and obj_match.re.groups > 1:
                version = obj_match.group(2)
        return version

    # Examine HTTP response.
    def examine_response(self, check_pattern, default_ver, version_pattern, response):
        self.utility.print_message(NOTE, 'Confirm string matching.')

        # Check exsisting contents.
        result = []
        if check_pattern != '*' and re.search(check_pattern, response, flags=re.IGNORECASE) is not None:
            result.append(True)
            # Check product version.
            result.append(self.check_version(default_ver, version_pattern, response))
        elif check_pattern == '*':
            result.append(True)
            # Check product version.
            result.append(self.check_version(default_ver, version_pattern, response))
        else:
            result.append(False)
            result.append(default_ver)
        return result

    # Explore unnecessary contents.
    def content_explorer(self, parsed, target_base, client):
        self.utility.print_message(NOTE, 'Explore unnecessary contents.')

        # Open signature file.
        signature_file = os.path.join(self.signature_dir, self.signature_file)
        product_list = []
        with codecs.open(signature_file, 'r', encoding='utf-8') as fin:
            signatures = fin.readlines()
            for idx, signature in enumerate(signatures):
                # Keep Alive to Metasploit per 10 count.
                if (idx + 1) % 10 == 0:
                    client.keep_alive()

                items = signature.replace('\n', '').replace('\r', '').split('@')
                product_name = items[0].lower()
                default_ver = items[1]
                path = items[2]
                check_pattern = items[3]
                version_pattern = items[4]
                target_url = ''
                if path.startswith('/') is True:
                    target_url = target_base + path[1:]
                else:
                    target_url = target_base + path

                # Get HTTP response (header + body).
                res, res_header, res_body = self.utility.send_request('GET', target_url)
                msg = '{}/{} Accessing : Status: {}, Url: {}'.format(idx + 1, len(signatures), res.status, target_url)
                self.utility.print_message(OK, msg)

                if res.status in [200, 301, 302]:
                    # Examine HTTP response.
                    result = self.examine_response(check_pattern, default_ver, version_pattern, res_header + res_body)
                    if result[0] is True:
                        # Adjust path value.
                        if path.endswith('/') is False:
                            path += '/'

                        # Get port number.
                        port_num = 80
                        if parsed.port is not None:
                            port_num = parsed.port

                        # Add product information.
                        product_list.extend([product_name + '@' + str(result[1]) + '@' + str(port_num) + '@' + path])
                        msg = 'Find product={}/{} from {}'.format(product_name, str(result[1]), target_url)
                        self.utility.print_message(OK, msg)

                time.sleep(self.delay_time)
        return product_list
