#!/usr/bin/python
# coding:utf-8
import os
import sys
import time
import configparser
from urllib.parse import urlparse
from Investigator import Investigate
from Recommender import Recommend
from util import Utilty


# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


# Display banner.
def show_banner(utility):
    banner = """
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
██████╗ ███████╗ ██████╗ ██████╗ ███╗   ███╗███╗   ███╗███████╗███╗   ██╗██████╗ ███████╗██████╗ 
██╔══██╗██╔════╝██╔════╝██╔═══██╗████╗ ████║████╗ ████║██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗
██████╔╝█████╗  ██║     ██║   ██║██╔████╔██║██╔████╔██║█████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝
██╔══██╗██╔══╝  ██║     ██║   ██║██║╚██╔╝██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗
██║  ██║███████╗╚██████╗╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║███████╗██║ ╚████║██████╔╝███████╗██║  ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝ (beta)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
""" + 'by ' + os.path.basename(__file__)
    utility.print_message(NONE, banner)
    show_credit(utility)
    time.sleep(utility.banner_delay)


# Show credit.
def show_credit(utility):
    credit = u"""
       =[ Recommender v0.0.1-beta                                            ]=
+ -- --=[ Author  : Isao Takaesu (@bbr_bbq)                                  ]=--
+ -- --=[ Website : https://github.com/13o-bbr-bbq/machine_learning_security ]=--
    """
    utility.print_message(NONE, credit)


if __name__ == "__main__":
    file_name = os.path.basename(__file__)
    full_path = os.path.dirname(os.path.abspath(__file__))
    utility = Utilty()

    # Read config.ini.
    config = configparser.ConfigParser()
    config.read(os.path.join(full_path, 'config.ini'))

    # Show banner.
    show_banner(utility)

    if len(sys.argv) == 1:
        utility.print_message(FAIL, 'Invalid parameter "{}"'.format(sys.argv))
        exit(1)

    if sys.argv[1] == 'TRAIN':
        utility.print_message(NOTE, 'Start {} mode.'.format(sys.argv[1]))
        obj_recommend = Recommend(utility)
        obj_recommend.training_model()
        utility.print_message(NOTE, 'End {} mode.'.format(sys.argv[1]))
    elif sys.argv[1] == 'TEST' and len(sys.argv) == 3:
        utility.print_message(NOTE, 'Start {} mode.'.format(sys.argv[1]))
        target_info = sys.argv[2]
        obj_parsed = urlparse(target_info)
        if utility.check_arg_value(obj_parsed.scheme, obj_parsed.netloc, str(obj_parsed.port), obj_parsed.path) is False:
            utility.print_message(FAIL, 'Invalid target: {}.'.format(target_info))
        else:
            # Generates feature vector
            obj_invest = Investigate(utility, target_info)
            all_feature_list, all_target_list = obj_invest.main_control()

            # recommends
            obj_recommend = Recommend(utility)
            for feature_list, target_list in zip(all_feature_list, all_target_list):
                flt_start = time.time()
                utility.print_message(NONE, '-'*130)
                utility.print_message(NOTE, 'Target url: {}\nParameter: {}'.format(target_list[0], target_list[1]))
                utility.print_message(OK, 'Feature: {}'.format(feature_list))
                obj_recommend.predict(feature_list)
                flt_elapsed_time = time.time() - flt_start
                utility.print_message(OK, 'Elapsed time :{}'.format(flt_elapsed_time) + '[sec]')

            utility.print_message(NOTE, 'End {} mode.'.format(sys.argv[1]))
    else:
        utility.print_message(FAIL, 'Invalid parameter: {}'.format(sys.argv))
