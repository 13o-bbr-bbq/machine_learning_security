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
    #fin = open('crawl_result\\crawl_result.json')
    dict_json = json.load(fin)
    lst_target = []
    for idx in range(len(dict_json)):
        items = dict_json[idx]['urls']
        for item in items:
            if str_allow_domain in item:
                lst_target.append(item)

    # generates feature vector
    obj_invest = Investigate(target)
    all_feature_list, all_target_list = obj_invest.main_control(lst_target)

    # recommends
    obj_recommend = Recommend()
    for feature_list, target_list in zip(all_feature_list, all_target_list):
        flt_start = time.time()
        print('target url: {0}, parameter: {1}'.format(target_list[0], target_list[1]))
        print('feature: {0}'.format(feature_list))
        obj_recommend.predict(feature_list)
        flt_elapsed_time = time.time() - flt_start
        print('Elapsed time  :{0}'.format(flt_elapsed_time) + "[sec]")
