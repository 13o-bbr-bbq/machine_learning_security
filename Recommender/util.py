#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import string
import random
import configparser
from datetime import datetime
from logging import getLogger, FileHandler, Formatter

# Printing colors.
OK_BLUE = '\033[94m'      # [*]
NOTE_GREEN = '\033[92m'   # [+]
FAIL_RED = '\033[91m'     # [-]
WARN_YELLOW = '\033[93m'  # [!]
ENDC = '\033[0m'
PRINT_OK = OK_BLUE + '[*]' + ENDC
PRINT_NOTE = NOTE_GREEN + '[+]' + ENDC
PRINT_FAIL = FAIL_RED + '[-]' + ENDC
PRINT_WARN = WARN_YELLOW + '[!]' + ENDC

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


# Utility class.
class Utilty:
    def __init__(self):
        # Read config.ini.
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        config.read(os.path.join(full_path, 'config.ini'))

        try:
            self.banner_delay = float(config['Common']['banner_delay'])
            self.report_date_format = config['Common']['date_format']
            self.con_timeout = float(config['Common']['con_timeout'])
            log_dir = os.path.join(full_path, config['Common']['log_path'])
            if os.path.exists(log_dir) is False:
                os.mkdir(log_dir)
            self.log_path = os.path.join(log_dir, config['Common']['log_file'])
        except Exception as e:
            self.print_message(FAIL, 'Reading config.ini is failure : {}'.format(e))
            sys.exit(1)

        # Setting logger.
        self.logger = getLogger('Recommender')
        self.logger.setLevel(20)
        file_handler = FileHandler(self.log_path)
        self.logger.addHandler(file_handler)
        formatter = Formatter('%(levelname)s,%(message)s')
        file_handler.setFormatter(formatter)

    # Print metasploit's symbol.
    def print_message(self, type, message):
        if os.name == 'nt':
            if type == NOTE:
                print('[+] ' + message)
            elif type == FAIL:
                print('[-] ' + message)
            elif type == WARNING:
                print('[!] ' + message)
            elif type == NONE:
                print(message)
            else:
                print('[*] ' + message)
        else:
            if type == NOTE:
                print(PRINT_NOTE + ' ' + message)
            elif type == FAIL:
                print(PRINT_FAIL + ' ' + message)
            elif type == WARNING:
                print(PRINT_WARN + ' ' + message)
            elif type == NONE:
                print(NOTE_GREEN + message + ENDC)
            else:
                print(PRINT_OK + ' ' + message)

    # Print exception messages.
    def print_exception(self, e, message):
        self.print_message(WARNING, 'type:{}'.format(type(e)))
        self.print_message(WARNING, 'args:{}'.format(e.args))
        self.print_message(WARNING, '{}'.format(e))
        self.print_message(WARNING, message)

    # Write logs.
    def write_log(self, loglevel, message):
        self.logger.log(loglevel, self.get_current_date() + ' ' + message)

    # Create random string.
    def get_random_token(self, length):
        chars = string.digits + string.ascii_letters
        return ''.join([random.choice(chars) for _ in range(length)])

    # Get current date.
    def get_current_date(self, indicate_format=None):
        if indicate_format is not None:
            date_format = indicate_format
        else:
            date_format = self.report_date_format
        return datetime.now().strftime(date_format)

    # Transform date from string to object.
    def transform_date_object(self, target_date, format=None):
        if format is None:
            return datetime.strptime(target_date, self.report_date_format)
        else:
            return datetime.strptime(target_date, format)

    # Transform date from object to string.
    def transform_date_string(self, target_date):
        return target_date.strftime(self.report_date_format)

    # Delete control character.
    def delete_ctrl_char(self, origin_text):
        clean_text = ''
        for char in origin_text:
            ord_num = ord(char)
            # Allow LF,CR,SP and symbol, character and numeric.
            if (ord_num == 10 or ord_num == 13) or (32 <= ord_num <= 126):
                clean_text += chr(ord_num)
        return clean_text

    # Check argument values.
    def check_arg_value(self, protocol, fqdn, port, path):
        # Check protocol.
        if protocol not in ['http', 'https']:
            self.print_message(FAIL, 'Invalid protocol : {}'.format(protocol))

        # Check IP address.
        if isinstance(fqdn, str) is False and isinstance(fqdn, int) is False:
            self.print_message(FAIL, 'Invalid IP address : {}'.format(fqdn))
            return False

        # Check port number.
        if port.isdigit() is False:
            self.print_message(FAIL, 'Invalid port number : {}'.format(port))
            return False
        elif (int(port) < 1) or (int(port) > 65535):
            self.print_message(FAIL, 'Invalid port number : {}'.format(port))
            return False

        # Check path.
        if isinstance(path, str) is False and isinstance(path, int) is False:
            self.print_message(FAIL, 'Invalid path : {}'.format(path))
            return False
        elif path.startswith('/') is False or path.endswith('/') is False:
            self.print_message(FAIL, 'Invalid path : {}'.format(path))
            return False

        return True
