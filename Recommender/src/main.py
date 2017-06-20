#!/usr/bin/python
# coding:utf-8
import sys
import time
import json
import urlparse
import ConfigParser
from datetime import datetime
from subprocess import Popen
from Investigator import Investigate
from Recommender import Recommend


def convert_feature_vector(dic_feature):
    lst_feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lst_feature[0] = dic_feature['Html']
    lst_feature[1] = dic_feature['Attribute']
    lst_feature[2] = dic_feature['JavaScript']
    lst_feature[3] = dic_feature['VBScript']
    lst_feature[4] = dic_feature['Quotation']
    lst_feature[5] = dic_feature['Double_quote']
    lst_feature[6] = dic_feature['Single_quote']
    lst_feature[7] = dic_feature['Back_quote']
    lst_feature[8] = dic_feature['Left']
    lst_feature[9] = dic_feature['Right']
    lst_feature[10] = dic_feature['Alert']
    lst_feature[11] = dic_feature['Prompt']
    lst_feature[12] = dic_feature['Confirm']
    lst_feature[13] = dic_feature['Backquote']
    lst_feature[14] = dic_feature['Start_script']
    lst_feature[15] = dic_feature['End_script']
    lst_feature[16] = dic_feature['Msgbox']
    return lst_feature

if __name__ == "__main__":
    inifile = ConfigParser.SafeConfigParser()
    try:
        inifile.read('config.ini')
    except:
        print('usage: can\'t read config.ini.')
        exit(1)
    output_base_path = inifile.get('Spider', 'output_base_path')
    output_filename = inifile.get('Spider', 'output_filename')
    spider_delay_time = inifile.get('Spider', 'delay_time')

    if len(sys.argv) == 2 and sys.argv[1] == 'TRAIN':
        obj_recommend = Recommend()
        obj_recommend.training_model()
        exit(0)

    target = sys.argv[1]
    obj_parsed = urlparse.urlparse(target)
    str_allow_domain = obj_parsed.netloc
    if obj_parsed.scheme != 'http' and obj_parsed.scheme != 'https':
        print('usage: Invalid scheme.')
        exit(1)

    # start Spider
    now_time = datetime.now().strftime('%Y%m%d%H%M%S')
    str_result_file = output_base_path + now_time + output_filename
    str_cmd_option = ' -a target_url=' + target + ' -a allow_domain=' + str_allow_domain +\
                     ' -a delay=' + spider_delay_time
    str_cmd = 'scrapy runspider Spider.py -o ' + str_result_file + str_cmd_option
    proc = Popen(str_cmd, shell=True)
    proc.wait()

    # get crawl's result
    fin = open(str_result_file)
    # DEBUG
    #fin = open('crawl_result\\20170616061147_crawl_result.json')
    dict_json = json.load(fin)
    lst_target = []
    for idx in range(len(dict_json)):
        items = dict_json[idx]['urls']
        for item in items:
            if str_allow_domain in item:
                lst_target.append(item)

    # generates feature vector
    obj_invest = Investigate(target)
    all_feature_list = obj_invest.main_control(lst_target)

    # recommends
    flt_start = time.time()
    obj_recommend = Recommend()

    for dic_feature in all_feature_list:
        print('feature: %s' % dic_feature)
        lst_feature = convert_feature_vector(dic_feature)
        obj_recommend.predict(lst_feature)

        flt_elapsed_time = time.time() - flt_start
        print("Elapsed time  :{0}".format(flt_elapsed_time) + "[sec]")
