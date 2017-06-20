#!/usr/bin/python
# coding:utf-8
import time
import random
import string
import copy
import re
import urlparse
import ConfigParser
import pandas as pd
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
        self.xss_signature = inifile.get('Investigator', 'xss_signature')
        self.scan_result = inifile.get('Investigator', 'scan_result')
        self.target_tags = inifile.get('Investigator', 'target_tags')
        self.scan_delay_time = float(inifile.get('Investigator', 'delay_time'))
        self.proxy_scheme = inifile.get('Investigator', 'proxy_scheme')
        self.proxy_addr = inifile.get('Investigator', 'proxy_addr')
        self.conn_timeout = inifile.get('Investigator', 'conn_timeout')
        self.delay_time = inifile.get('Investigator', 'delay_time')
        obj_parsed = urlparse.urlparse(target)
        self.str_allow_domain = obj_parsed.netloc
        self.obj_signatures = pd.read_csv(self.xss_signature, encoding='utf-8').fillna('')

        # dictionary of feature
        self.dic_feature = {'Url': '', 'Param': '', 'Html': 0, 'Attribute': 0, 'JavaScript': 0, 'VBScript': 0,
                            'Quotation': 0, 'Double_quote': 0, 'Single_quote': 0, 'Back_quote': 0,
                            'Left': 0, 'Right': 0, 'Alert': 0, 'Prompt': 0, 'Confirm': 0, 'Backquote': 0,
                            'Start_script': 0, 'End_script': 0, 'Msgbox': 0}
        self.dic_feature_temp = {}

    def gen_rand_str(self, int_length):
        str_chars = string.digits + string.letters
        return ''.join([random.choice(str_chars) for idx in range(int_length)])

    def convert_feature_to_vector(self, str_type, str_feature):
        if str_type == 'html':
            if str_feature == '':
                return 0
            elif str_feature == 'comment':
                return 1
            elif str_feature == 'body':
                return 2
            elif str_feature == 'frame':
                return 3
            elif str_feature == 'img':
                return 4
            elif str_feature == 'input':
                return 5
            elif str_feature == 'script':
                return 6
            elif str_feature == 'textarea':
                return 7
            elif str_feature == 'iframe':
                return 8
            elif str_feature == 'a':
                return 9
            elif str_feature == 'div':
                return 10
            else:
                return 99
        elif str_type == 'attribute':
            if str_feature == '':
                return 0
            elif str_feature == 'id':
                return 1
            elif str_feature == 'src':
                return 2
            elif str_feature == 'value':
                return 3
            elif str_feature == 'href':
                return 4
            else:
                return 99
        elif str_type == 'JavaScript':
            if str_feature == '':
                return 0
            elif str_feature == '/*':
                return 1
            elif str_feature == '//':
                return 2
            elif str_feature == 'var':
                return 3
            else:
                return 9
        elif str_type == 'VBScript':
            if str_feature == '':
                return 0
            elif str_feature == 'plane':
                return 1
            else:
                return 9
        elif str_type == 'quotation':
            if str_feature == '':
                return 0
            elif str_feature == '"':
                return 1
            elif str_feature == "'":
                return 2
            else:
                return 9

    def get_escape_key_name(self, str_mark):
        if str_mark == '"':
            return 'Double_quote'
        elif str_mark == "'":
            return 'Single_quote'
        elif str_mark == '`':
            return 'Back_quote'
        elif str_mark == '<':
            return 'Left'
        elif str_mark == '>':
            return 'Right'
        elif str_mark == 'alert();':
            return 'Alert'
        elif str_mark == 'prompt();':
            return 'Prompt'
        elif str_mark == 'confirm();':
            return 'Confirm'
        elif str_mark == '``':
            return 'Backquote'
        elif str_mark == '<script>':
            return 'Start_script'
        elif str_mark == '</script>':
            return 'End_script'
        elif str_mark == 'Msgbox':
            return 'Msgbox'

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
        obj_bs = BeautifulSoup(str_response)
        lst_get_tags = self.target_tags.split(',')
        for str_tag in lst_get_tags:
            obj_bs_temp = copy.copy(obj_bs)
            lst_tags = obj_bs_temp.find_all(str_tag.replace(' ', ''))
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
                        lst_tag_feature.append(dic_feature_attr)
                # checking contents
                if str_craft_value in obj_tag.get_text():
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
                    lst_tag_feature.append(dic_feature_contents)
        return lst_tag_feature

    def main_control(self, lst_target):
        all_feature_list = []
        for str_url in lst_target:
            obj_parsed = urlparse.urlparse(str_url)
            # checking domain
            if self.str_allow_domain != obj_parsed.netloc:
                continue

            # checking parameters
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
                else:
                    continue
            time.sleep(float(self.delay_time))
        return all_feature_list
