#!/usr/bin/python
# coding:utf-8
import time
import random
import string
import copy
import json
import re
import urlparse
import ConfigParser
import pandas as pd
from datetime import datetime
from subprocess import Popen
from requests import Request, Session
from bs4 import BeautifulSoup


class Investigate:
    def __init__(self, target):
        inifile = ConfigParser.SafeConfigParser()
        try:
            inifile.read('config.ini')
        except:
            print('usage: can\'t read config.ini.')
            exit(1)
        self.output_base_path = inifile.get('Spider', 'output_base_path')
        self.output_filename = inifile.get('Spider', 'output_filename')
        self.spider_delay_time = inifile.get('Spider', 'delay_time')
        self.xss_signature = inifile.get('Investigator', 'xss_signature')
        self.scan_result = inifile.get('Investigator', 'scan_result')
        self.target_tags = inifile.get('Investigator', 'target_tags')
        self.convert_tags = inifile.get('Investigator', 'convert_tags')
        self.convert_attr = inifile.get('Investigator', 'convert_attr')
        self.convert_js = inifile.get('Investigator', 'convert_js')
        self.convert_vbs = inifile.get('Investigator', 'convert_vbs')
        self.convert_quot = inifile.get('Investigator', 'convert_quot')
        self.output_key = inifile.get('Investigator', 'output_key')
        self.escape_key = inifile.get('Investigator', 'escape_key')
        self.escape_value = inifile.get('Investigator', 'escape_value')
        self.scan_delay_time = float(inifile.get('Investigator', 'delay_time'))
        self.proxy_scheme = inifile.get('Investigator', 'proxy_scheme')
        self.proxy_addr = inifile.get('Investigator', 'proxy_addr')
        self.conn_timeout = inifile.get('Investigator', 'conn_timeout')
        self.delay_time = inifile.get('Investigator', 'delay_time')
        self.str_target_url = target
        obj_parsed = urlparse.urlparse(target)
        self.str_allow_domain = obj_parsed.netloc
        self.obj_signatures = pd.read_csv(self.xss_signature, encoding='utf-8').fillna('')

        # dictionary of feature(base)
        self.dic_feature = {'Url': '', 'Param': ''}
        self.dic_feature_temp = {}

        # make feature dictionary and convert table
        self.dic_convert_output = {}
        self.dic_convert_escape = {}
        self.make_dictionary()

    def make_dictionary(self):
        # make conversion table for output places
        list_tags = self.convert_tags.split(',')
        list_attr = self.convert_attr.split(',')
        list_js = self.convert_js.split(',')
        list_vbs = self.convert_vbs.split(',')
        list_quot = self.convert_quot.split(',')
        for idx, str_tag in enumerate(list_tags):
            self.dic_convert_output[str_tag] = idx
        for idx, str_attr in enumerate(list_attr):
            self.dic_convert_output[str_attr] = idx
        for idx, str_js in enumerate(list_js):
            self.dic_convert_output[str_js] = idx
        for idx, str_vbs in enumerate(list_vbs):
            self.dic_convert_output[str_vbs] = idx
        for idx, str_quot in enumerate(list_quot):
            self.dic_convert_output[str_quot] = idx

        # make conversion table for escape
        list_esc_value = self.escape_value.split(',')
        list_esc_key = self.escape_key.split(',')
        for str_key, str_value in zip(list_esc_value, list_esc_key):
            self.dic_convert_escape[str_key] = str_value

        # make feature dictionary
        list_output = self.output_key.split(',')
        for str_output in list_output:
            self.dic_feature[str_output] = 0
        for str_escape in list_esc_key:
            self.dic_feature[str_escape] = 0

    def convert_feature_list(self, dic_feature):
        lst_feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        list_output = self.output_key.split(',')
        list_esc_key = self.escape_key.split(',')
        idx = 0
        for str_output_key in list_output:
            lst_feature[idx] = dic_feature[str_output_key]
            idx += 1
        for str_escape_key in list_esc_key:
            lst_feature[idx] = dic_feature[str_escape_key]
            idx += 1
        return lst_feature

    def gen_rand_str(self, int_length):
        str_chars = string.digits + string.letters
        return ''.join([random.choice(str_chars) for idx in range(int_length)])

    def convert_feature_to_vector(self, str_type, str_feature):
        if str_feature == '':
            str_feature = 'none_' + str_type
        try:
            return int(self.dic_convert_output[str_feature])
        except:
            return 99

    def get_escape_key_name(self, str_mark):
        try:
            return self.dic_convert_escape[str_mark]
        except:
            print('usage: no conversion key.')
            exit(1)

    def specify_escape_type(self, str_response, str_seek_before, str_seek_after, str_signature, dic_feature_local):
        str_pattern = r'.*' + str_seek_before + r'(.*)' + str_seek_after
        obj_match = re.match(str_pattern, str_response.replace('\t', '').replace('\r', '').replace('\n', ''))
        if obj_match is not None:
            lst_signatures = str_signature.split('|')
            for str_mark in lst_signatures:
                if str_mark in obj_match.group(1):
                    dic_feature_local[self.get_escape_key_name(str_mark)] = 0
                else:
                    dic_feature_local[self.get_escape_key_name(str_mark)] = 1
        return dic_feature_local

    def specify_feature_escape(self, dict_params, int_idx, str_url, str_param, dic_feature_local):
        dict_craft_params = copy.deepcopy(dict_params)
        str_seek_before = self.gen_rand_str(3)
        str_seek_after = self.gen_rand_str(3)
        str_signature = str(self.obj_signatures['Signature'][int_idx])
        lst_signatures = str_signature.split('|')
        str_inspect_value = ''
        for elem in lst_signatures:
            str_inspect_value += elem
        str_inspect = str_seek_before + dict_craft_params[str_param][0] + str_inspect_value + str_seek_after
        dict_craft_params[str_param] = str_inspect
        obj_session = Session()
        obj_request = Request('GET', str_url, params=dict_craft_params)
        obj_prepped = obj_session.prepare_request(obj_request)
        obj_response = None
        try:
            obj_response = obj_session.send(obj_prepped,
                                            verify=True,
                                            proxies={self.proxy_scheme: self.proxy_addr},
                                            timeout=int(self.conn_timeout),
                                            allow_redirects=False
                                            )
        except:
            print('usage: timeout. ' + str_url)
            return dic_feature_local
        return self.specify_escape_type(obj_response.text,
                                        str_seek_before,
                                        str_seek_after,
                                        str_signature,
                                        dic_feature_local)

    def specify_feature(self, str_response, str_url, dict_params, str_param, str_craft_value):
        lst_tag_feature = []
        obj_bs = BeautifulSoup(str_response, 'lxml')
        lst_get_tags = self.target_tags.split(',')
        for str_tag in lst_get_tags:
            obj_bs_temp = copy.copy(obj_bs)
            lst_tags = obj_bs_temp.find_all(str_tag)
            obj_bs_temp = None
            if len(lst_tags) == 0:
                continue

            for obj_tag in lst_tags:
                # checking attributes
                lst_attrs = obj_tag.attrs.keys()
                for str_attr in lst_attrs:
                    # including inspection string in attribute value?
                    if str_craft_value in obj_tag.attrs[str_attr]:
                        str_tag = str(obj_tag)
                        idx = str(obj_tag).find(str_craft_value)
                        # convert feature vector for "output place"
                        dic_feature_attr = copy.deepcopy(self.dic_feature)
                        dic_feature_attr['Html'] = self.convert_feature_to_vector('html', obj_tag.name)
                        dic_feature_attr['Attribute'] = self.convert_feature_to_vector('attribute', str_attr)
                        dic_feature_attr['JavaScript'] = 0
                        dic_feature_attr['VBScript'] = 0
                        dic_feature_attr['Quotation'] = self.convert_feature_to_vector('quotation', str_tag[idx-1:idx])

                        # convert feature vector for "escape"
                        for idx in range(len(self.obj_signatures)):
                            dic_feature_attr = self.specify_feature_escape(dict_params,
                                                                           idx,
                                                                           str_url,
                                                                           str_param,
                                                                           dic_feature_attr)
                        lst_tag_feature.append(self.convert_feature_list(dic_feature_attr))
                # checking contents
                if str_craft_value in obj_tag.get_text():
                    # TODO: checking output values in HTML comment syntax
                    # convert feature vector for "output place"
                    dic_feature_contents = copy.deepcopy(self.dic_feature)
                    dic_feature_contents['Html'] = self.convert_feature_to_vector('html', obj_tag.name)
                    dic_feature_contents['Attribute'] = 0
                    dic_feature_contents['JavaScript'] = 0
                    dic_feature_contents['VBScript'] = 0
                    dic_feature_contents['Quotation'] = 0

                    # convert feature vector for "escape"
                    for idx in range(len(self.obj_signatures)):
                        dic_feature_contents = self.specify_feature_escape(dict_params,
                                                                           idx,
                                                                           str_url,
                                                                           str_param,
                                                                           dic_feature_contents)
                    lst_tag_feature.append(self.convert_feature_list(dic_feature_contents))
        return lst_tag_feature

    def run_spider(self):
        now_time = datetime.now().strftime('%Y%m%d%H%M%S')
        str_result_file = self.output_base_path + now_time + self.output_filename
        str_cmd_option = ' -a target_url=' + self.str_target_url + ' -a allow_domain=' + self.str_allow_domain + \
                         ' -a delay=' + self.spider_delay_time
        str_cmd = 'scrapy runspider Spider.py -o ' + str_result_file + str_cmd_option
        proc = Popen(str_cmd, shell=True)
        proc.wait()

        # get crawl's result
        fin = open(str_result_file)
        # fin = open('crawl_result\\crawl_result.json')
        dict_json = json.load(fin)
        lst_target = []
        for idx in range(len(dict_json)):
            items = dict_json[idx]['urls']
            for item in items:
                if self.str_allow_domain in item:
                    lst_target.append(item)
        return lst_target

    def main_control(self):
        # start Spider
        lst_target = self.run_spider()

        # start Investigation
        all_feature_list = []
        all_target_list = []
        for str_url in lst_target:
            obj_parsed = urlparse.urlparse(str_url)
            # checking domain
            if self.str_allow_domain != obj_parsed.netloc:
                continue

            # checking parameters(query parameters only)
            if '?' in str_url:
                dict_params = urlparse.parse_qs(str_url[str_url.find('?') + 1:])
                lst_param = dict_params.keys()
            else:
                continue

            # checking each parameter
            for str_param in lst_param:
                # checking reflected value at HTTP response
                dict_craft_params = copy.deepcopy(dict_params)
                str_value = dict_params[str_param][0]
                str_seek_before = self.gen_rand_str(3)
                str_seek_after = self.gen_rand_str(3)
                dict_craft_params[str_param] = str_seek_before + str_value + str_seek_after
                str_target = str_url[:str_url.find('?')]
                obj_session = Session()
                obj_request = Request('GET', str_target, params=dict_craft_params)
                obj_prepped = obj_session.prepare_request(obj_request)
                obj_response = None
                try:
                    obj_response = obj_session.send(obj_prepped,
                                                    verify=True,
                                                    proxies={self.proxy_scheme: self.proxy_addr},
                                                    timeout=int(self.conn_timeout),
                                                    allow_redirects=False
                                                    )
                except:
                    print('usage: timeout. ' + str_url)
                    continue

                print(str_url + ',' + str(obj_response.status_code))
                if dict_craft_params[str_param] in obj_response.text:
                    # input URL and parameter to feature dictionary
                    self.dic_feature['Url'] = str_url
                    self.dic_feature['Param'] = str_param

                    # investigate output place and escape type
                    feature_list = self.specify_feature(obj_response.text,
                                                        str_target,
                                                        dict_params,
                                                        str_param,
                                                        dict_craft_params[str_param])
                    for feature in feature_list:
                        all_feature_list.append(feature)
                        all_target_list.append([str_url, str_param])
                else:
                    continue
            time.sleep(float(self.delay_time))
        return all_feature_list, all_target_list
