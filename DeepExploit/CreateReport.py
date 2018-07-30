#!/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import codecs
import glob
import configparser
import pandas as pd
from datetime import datetime
from docopt import docopt
from jinja2 import Environment, FileSystemLoader
from util import Utilty

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.


# Create report.
class CreateReport:
    def __init__(self):
        self.util = Utilty()

        # Read config file.
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(full_path, 'config.ini'))
        except Exception as err:
            self.util.print_exception(err, 'File exists error')
            sys.exit(1)

        self.report_date_format = config['Report']['date_format']
        self.report_test_path = os.path.join(full_path, config['Report']['report_test'])
        self.report_test_file = os.path.join(self.report_test_path, config['Report']['report_test_file'])
        self.template_test = config['Report']['template_test']
        self.report_train_path = os.path.join(self.report_test_path, config['Report']['report_train'])
        self.report_train_file = os.path.join(self.report_train_path, config['Report']['report_train_file'])
        self.template_train = config['Report']['template_train']
        self.header_train = str(config['Report']['header_train']).split('@')
        self.header_test = str(config['Report']['header_test']).split('@')

    def create_report(self, mode='train', start_date=None):
        # Check mode.
        if mode not in ['train', 'test']:
            self.util.print_message(FAIL, 'Invalid mode: {}'.format(mode))
            exit(1)

        # Gather reporting items.
        if mode == 'train':
            self.util.print_message(NOTE, 'Creating training report.')
            csv_file_list = glob.glob(os.path.join(self.report_train_path, '*.csv'))

            # Create DataFrame.
            content_list = []
            for file in csv_file_list:
                df = pd.read_csv(file, names=self.header_train, sep=',')
                df['date'] = pd.to_datetime(df['date'])
                selected_df = df[(start_date < df['date'])]
                content_list.append(selected_df)

            if len(content_list) != 0:
                df_csv = pd.concat(content_list).drop_duplicates().sort_values(by=['ip', 'port'],
                                                                               ascending=True).reset_index(drop=True,
                                                                                                           col_level=1)

                items = []
                for idx in range(len(df_csv)):
                    items.append({'ip_addr': df_csv.loc[idx, 'ip'],
                                  'port': df_csv.loc[idx, 'port'],
                                  'prod_name': df_csv.loc[idx, 'service'],
                                  'vuln_name': df_csv.loc[idx, 'vuln_name'],
                                  'description': df_csv.loc[idx, 'description'],
                                  'type': df_csv.loc[idx, 'type'],
                                  'exploit': df_csv.loc[idx, 'exploit'],
                                  'target': df_csv.loc[idx, 'target'],
                                  'payload': df_csv.loc[idx, 'payload'],
                                  'ref': str(df_csv.loc[idx, 'reference']).replace('@', '<br>')})

                try:
                    # Setting template.
                    env = Environment(loader=FileSystemLoader(self.report_train_path))
                    template = env.get_template(self.template_train)
                    pd.set_option('display.max_colwidth', -1)
                    html = template.render({'title': 'Deep Exploit Scan Report', 'items': items})

                    # Write report.
                    with codecs.open(self.report_train_file, 'w', 'utf-8') as fout:
                        fout.write(html)
                except Exception as err:
                    self.util.print_exception(err, 'Creating report error.')
            else:
                self.util.print_message(WARNING, 'Exploitation result is not found.')
            self.util.print_message(OK, 'Creating training report done.')
        else:
            self.util.print_message(NOTE, 'Creating testing report.')
            csv_file_list = glob.glob(os.path.join(self.report_test_path, '*.csv'))

            # Create DataFrame.
            content_list = []
            for file in csv_file_list:
                df = pd.read_csv(file, names=self.header_test, sep=',')
                df['date'] = pd.to_datetime(df['date'])
                selected_df = df[(start_date < df['date'])]
                content_list.append(selected_df)

            if len(content_list) != 0:
                df_csv = pd.concat(content_list).drop_duplicates().sort_values(by=['ip', 'port'],
                                                                               ascending=True).reset_index(drop=True,
                                                                                                           col_level=1)

                items = []
                for idx in range(len(df_csv)):
                    items.append({'ip_addr': df_csv.loc[idx, 'ip'],
                                  'port': df_csv.loc[idx, 'port'],
                                  'source_ip_addr': df_csv.loc[idx, 'src_ip'],
                                  'prod_name': df_csv.loc[idx, 'service'],
                                  'vuln_name': df_csv.loc[idx, 'vuln_name'],
                                  'description': df_csv.loc[idx, 'description'],
                                  'type': df_csv.loc[idx, 'type'],
                                  'exploit': df_csv.loc[idx, 'exploit'],
                                  'target': df_csv.loc[idx, 'target'],
                                  'payload': df_csv.loc[idx, 'payload'],
                                  'ref': str(df_csv.loc[idx, 'reference']).replace('@', '<br>')})

                try:
                    # Setting template.
                    env = Environment(loader=FileSystemLoader(self.report_test_path))
                    template = env.get_template(self.template_test)
                    pd.set_option('display.max_colwidth', -1)
                    html = template.render({'title': 'Deep Exploit Scan Report', 'items': items})

                    # Write report.
                    with codecs.open(self.report_test_file, 'w', 'utf-8') as fout:
                        fout.write(html)
                except Exception as err:
                    self.util.print_exception(err, 'Creating report error.')
            else:
                self.util.print_message(WARNING, 'Exploitation result is not found.')
            self.util.print_message(OK, 'Creating testing report done.')


# Define command option.
__doc__ = """{f}
Usage:
    {f} (-m <mode> | --mode <mode>) [(-s <start> | --start <start>)]
    {f} -h | --help

Options:
    -m --mode     Require  : Creating mode "train/test".
    -s --start    Optional : begining start time (format='%Y%m%d%H%M%S')
    -h --help     Optional : Show this screen and exit.
""".format(f=__file__)


# Parse command arguments.
def command_parse():
    args = docopt(__doc__)
    mode = args['<mode>']
    start_time = args['<start>']
    return mode, start_time


if __name__ == '__main__':
    # Get command arguments.
    mode, start_time = command_parse()

    # Create report.
    report = CreateReport()
    try:
        if start_time is None:
            start_time = '19000101000000'
        get_date = datetime.strptime(start_time, '%Y%m%d%H%M%S')
        report.create_report(mode, pd.to_datetime(report.util.transform_date_string(get_date)))
    except Exception as err:
        report.util.print_exception(err, 'Invalid date format: {}.'.format(start_time))
        exit(1)
