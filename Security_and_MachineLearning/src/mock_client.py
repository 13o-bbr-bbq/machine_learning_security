#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import tornado.httpclient
from logging import getLogger, FileHandler, StreamHandler, Formatter

MAX_COUNT = 300
REQ_MODE = 'rapidly'
WAIT_TIME = 0.5
MODE_N = 'normal'
MODE_A = 'attack'
MODE_L = 'load'
DIR = os.path.dirname(os.path.abspath(__file__))
LOG_NAME = os.path.join(DIR, 'server_response.csv')
TARGET = 'http://127.0.0.1:8888/?param='

if __name__ == '__main__':
    # Setting logger.
    logger = getLogger('LoggingMockClient')
    logger.setLevel(20)
    fh = FileHandler(LOG_NAME)
    logger.addHandler(fh)
    sh = StreamHandler()
    logger.addHandler(sh)
    formatter = Formatter('%(asctime)s,%(levelname)s,%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # Pseudo scan.
    value = ''
    for idx in range(MAX_COUNT):
        if REQ_MODE == 'complex':
            if 0 <= idx < 50:
                value = MODE_N
            elif 50 <= idx < 100:
                value = MODE_A
            elif 100 <= idx < 200:
                value = MODE_N
            elif 200 <= idx:
                value = MODE_L
        elif REQ_MODE == 'rapidly':
            if 0 <= idx < 100:
                value = MODE_N
            else:
                value = MODE_A
        elif REQ_MODE == 'slowly':
            if 0 <= idx < 100:
                value = MODE_N
            else:
                value = MODE_L
        else:
            print('Invalid mode.')
            sys.exit(1)

        # Send request, measure response time.
        http_client = tornado.httpclient.HTTPClient()
        start = time.perf_counter()
        response = http_client.fetch(TARGET + value)
        elapsed_time = time.perf_counter() - start
        http_client.close()

        # Logging.
        logger.log(20, 'Response_time:' + str(elapsed_time))

        time.sleep(WAIT_TIME)

    print('finish!!')
