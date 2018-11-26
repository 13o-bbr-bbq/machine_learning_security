#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import string
import random
import codecs
import json
import urllib3
import configparser
from urllib3 import util
from datetime import datetime
from subprocess import Popen
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

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
        try:
            config.read(os.path.join(full_path, 'config.ini'))
        except FileExistsError as err:
            self.print_message(FAIL, 'File exists error: {}'.format(err))
            sys.exit(1)

        # Utility setting value.
        self.http_timeout = float(config['Utility']['http_timeout'])
        self.max_target_url = int(config['Utility']['max_target_url'])
        self.max_target_byte = int(config['Utility']['max_target_byte'])
        if int(config['Utility']['scramble']) == 1:
            self.is_scramble = True

        # Report setting value.
        self.report_date_format = str(config['Report']['date_format'])

        # Spider setting value.
        self.output_base_path = config['Spider']['output_base_path']
        self.store_path = os.path.join(full_path, self.output_base_path)
        if os.path.exists(self.store_path) is False:
            os.mkdir(self.store_path)
        self.output_filename = config['Spider']['output_filename']
        self.spider_concurrent_reqs = config['Spider']['concurrent_reqs']
        self.spider_depth_limit = config['Spider']['depth_limit']
        self.spider_delay_time = config['Spider']['delay_time']
        self.spider_time_out = config['Spider']['time_out']
        self.spider_item_count = config['Spider']['item_count']
        self.spider_page_count = config['Spider']['page_count']
        self.spider_error_count = config['Spider']['error_count']

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
    def transform_date_object(self, target_date):
        return datetime.strptime(target_date, self.report_date_format)

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

    # Check web port.
    def check_web_port(self, target_ip, port_list, client):
        self.print_message(NOTE, 'Check web port.')
        web_port_list = []
        for port_num in port_list:
            # Send HTTP request.
            http = urllib3.PoolManager(timeout=self.http_timeout)
            for scheme in ['http://', 'https://']:
                target_url = scheme + target_ip + ':' + port_num
                try:
                    client.keep_alive()
                    self.print_message(OK, 'Target URL: {}'.format(target_url))
                    res = http.request('GET', target_url)
                    self.print_message(OK, 'Port "{}" is web port. status={}'.format(port_num, res.status))
                    web_port_list.append([port_num, scheme])
                    break
                except Exception as e:
                    self.print_message(WARNING, 'Port "{}" is not web port.'.format(port_num))
        return web_port_list

    # Send http request.
    def send_request(self, method, target_url):
        res_header = ''
        res_body = ''
        res = None
        http = urllib3.PoolManager(timeout=self.http_timeout)
        try:
            res = http.request(method, target_url)
            for header in res.headers.items():
                res_header += header[0] + ': ' + header[1] + '\r\n'
            res_body = '\r\n\r\n' + res.data.decode('utf-8')
        except Exception as e:
            self.print_exception(e, 'Access is failure : {}'.format(target_url))
        return res, res_header, res_body

    # Parse url.
    def parse_url(self, url):
        parsed = None
        try:
            parsed = util.parse_url(url)
        except Exception as e:
            self.util.print_exception(e, 'Parsed error : {}'.format(url))
        return parsed

    # Running spider.
    def run_spider(self, target_ip, target_web, client):
        # Execute crawling using Scrapy.
        all_targets_log = []
        for target_info in target_web:
            target_url = target_info[1] + target_ip + ':' + target_info[0] + '/'
            target_log = [target_url]
            response_log = target_ip + '_' + target_info[0] + '.log'
            now_time = self.get_current_date('%Y%m%d%H%M%S')
            result_file = os.path.join(self.output_base_path, now_time + self.output_filename)
            option = ' -a target_url=' + target_url + ' -a allow_domain=' + target_ip + \
                     ' -a concurrent=' + self.spider_concurrent_reqs + ' -a depth_limit=' + self.spider_depth_limit + \
                     ' -a delay=' + self.spider_delay_time + ' -a store_path=' + self.store_path + \
                     ' -a response_log=' + response_log + ' -a msgrpc_host=' + client.host + \
                     ' -a msgrpc_port=' + str(client.port) + ' -a msgrpc_token=' + client.token.decode('utf-8') + \
                     ' -a msgrpc_console_id=' + client.console_id.decode('utf-8') + ' -o ' + result_file
            close_opton = ' -s CLOSESPIDER_TIMEOUT=' + self.spider_time_out + \
                          ' -s CLOSESPIDER_ITEMCOUNT=' + self.spider_item_count + \
                          ' -s CLOSESPIDER_PAGECOUNT=' + self.spider_page_count + \
                          ' -s CLOSESPIDER_ERRORCOUNT=' + self.spider_error_count + ' '
            command = 'scrapy runspider' + close_opton + 'Spider.py' + option
            proc = Popen(command, shell=True)
            proc.wait()

            # Get crawling result.
            dict_json = {}
            if os.path.exists(result_file):
                with codecs.open(result_file, 'r', encoding='utf-8') as fin:
                    target_text = self.delete_ctrl_char(fin.read())
                    if target_text != '':
                        dict_json = json.loads(target_text)
                    else:
                        self.print_message(WARNING, '[{}] is empty.'.format(result_file))

            # Exclude except allowed domains.
            for idx in range(len(dict_json)):
                items = dict_json[idx]['urls']
                for item in items:
                    try:
                        if target_ip == util.parse_url(item).host:
                            target_log.append(item)
                    except Exception as err:
                        self.print_exception(err, 'Parsed error: {}'.format(item))
            all_targets_log.append([target_url, os.path.join(self.store_path, response_log), list(set(target_log))])
        return all_targets_log
