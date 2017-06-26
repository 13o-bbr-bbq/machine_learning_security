#!/usr/bin/python
# coding:utf-8
import sys
import time
import urlparse
from Investigator import Investigate
from Recommender import Recommend


if __name__ == "__main__":
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

    # generates feature vector
    obj_invest = Investigate(target)
    all_feature_list, all_target_list = obj_invest.main_control()

    # recommends
    obj_recommend = Recommend()
    for feature_list, target_list in zip(all_feature_list, all_target_list):
        flt_start = time.time()
        print('target url: {0}, parameter: {1}'.format(target_list[0], target_list[1]))
        print('feature: {0}'.format(feature_list))
        obj_recommend.predict(feature_list)
        flt_elapsed_time = time.time() - flt_start
        print('Elapsed time  :{0}'.format(flt_elapsed_time) + "[sec]")
