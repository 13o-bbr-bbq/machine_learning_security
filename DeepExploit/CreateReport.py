#!/bin/env python
# -*- coding: utf-8 -*-
import sys
import glob
import configparser
import pandas as pd
from jinja2 import Environment, FileSystemLoader


# Create report.
class CreateReport:
    def __init__(self):
        # Read config file.
        config = configparser.ConfigParser()
        try:
            config.read('./config.ini')
        except FileExistsError as err:
            print('File exists error: {0}', err)
            sys.exit(1)

        self.report_path = config['Report']['report_path']
        self.report_name = config['Report']['report_name']
        self.template = config['Report']['template']
        self.header = str(config['Report']['header']).split('@')

    def create_report(self):
        # Gather reporting items.
        csv_file_list = glob.glob(self.report_path + '*.csv')

        # Create DataFrame.
        content_list = []
        for file in csv_file_list:
            content_list.append(pd.read_csv(file, names=self.header, sep=','))
        df_csv = pd.concat(content_list).drop_duplicates().sort_values(by=['ip', 'port'], ascending=True).reset_index(drop=True, col_level=1)

        items = []
        for idx in range(len(df_csv)):
            items.append({'ip_addr': df_csv.loc[idx, 'ip'],
                          'port': df_csv.loc[idx, 'port'],
                          'prod_name': df_csv.loc[idx, 'service'],
                          'vuln_name': df_csv.loc[idx, 'vuln_name'],
                          'type': df_csv.loc[idx, 'type'],
                          'description': df_csv.loc[idx, 'description'],
                          'exploit': df_csv.loc[idx, 'exploit'],
                          'target': df_csv.loc[idx, 'target'],
                          'payload': df_csv.loc[idx, 'payload'],
                          'ref': str(df_csv.loc[idx, 'reference']).replace('@', '<br>')})

        # Setting template.
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template(self.template)
        pd.set_option('display.max_colwidth', -1)
        html = template.render({'title': 'GyoiThon Scan Report', 'items': items})
        with open(self.report_path + self.report_name, 'w') as fout:
            fout.write(html)


if __name__ == '__main__':
    report = CreateReport()
    report.create_report()
    print('Finish!!')
