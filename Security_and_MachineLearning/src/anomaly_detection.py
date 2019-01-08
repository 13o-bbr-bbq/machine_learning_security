#!/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import datetime
import codecs
import re
import numpy as np
from stat import *
from sklearn.neighbors import NearestNeighbors

# Anomaly detection parameters.
WINDOW_SIZE = 10
RATE_WARNING = 0.5
RATE_ERROR = 0.8
MSG_WARNING = 'Warning'
MSG_ERROR = 'Error'
K = 1

# Surveillance directory and file.
DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_LOG = 'server_response.csv'

# Regulation of Window slide.
max_count = 0
old_count = 0


# Monitor log files.
def watch():
    # Get first timestamp.
    timestamp = time.mktime(datetime.datetime.now().utctimetuple())
    while True:
        for file in os.listdir(DIR):
            if TARGET_LOG in file:
                # Check updated date of target log.
                target_file = os.path.join(DIR, file)
                file_timestamp = os.stat(target_file)[ST_MTIME]
                if timestamp < file_timestamp:
                    # Update timestamp.
                    timestamp = file_timestamp

                    # Read log.
                    fin = codecs.open(target_file, encoding='utf-8')
                    content = fin.read()
                    fin.close()

                    # Judgement.
                    check_logs(content)
        # Wait.
        time.sleep(0.5)


# Judgement of anomaly detection.
def check_logs(content):
    global max_count
    global old_count
    # Extract response time from log.
    all_res_time = re.findall('Response_time\:(\d{1,100}\.\d{1,100})*', content)
    if len(all_res_time) == 0:
        return

    all_res_time = list(map(float, all_res_time))
    response_time = all_res_time[old_count:]
    response_time = list(map(float, response_time))
    max_count = len(all_res_time)

    # Anomaly detection
    if len(all_res_time) > 0:
        # Abnormal part detection.
        anomaly_detection(response_time)


# Select log step by window size.
def emb(lst, dim):
    emb = np.empty((0, dim), float)
    for idx in range(len(lst) - dim + 1):
        tmp = np.array(lst[idx: idx + dim])[::-1].reshape((1, -1))
        emb = np.append(emb, tmp, axis=0)
    return emb


# Anomaly detection using K-NearestNeighbors.
def anomaly_detection(res_time):
    global max_count
    global old_count
    if len(res_time) < WINDOW_SIZE * 2:
        return

    # Get train data.
    train_data = res_time[0: WINDOW_SIZE]
    train = emb(train_data, WINDOW_SIZE)

    # Get test data.
    test_data = res_time[len(train_data): max_count]
    test = emb(test_data, WINDOW_SIZE)
    old_count += WINDOW_SIZE

    # Fit.
    clf = NearestNeighbors(n_neighbors=K)
    clf.fit(train)

    # Calculate distances.
    distances = clf.kneighbors(test)[0]

    # Judge anomaly detection.
    for d in distances:
        now_date = str(datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3])
        print('Distance: {}'.format(d[0]))
        if RATE_WARNING <= d[0] < RATE_ERROR:
            ''' Any notice. ex) posting slack, sending email and terminate scanner. '''
            print('WARNING: Detact anomaly!! Now date: {}'.format(now_date))
        elif RATE_ERROR <= d[0]:
            ''' Any notice. ex) posting slack, sending email and terminate scanner. '''
            print('ERROR: Detact anomaly!! Now date: {}'.format(now_date))


if __name__ == '__main__':
    watch()
