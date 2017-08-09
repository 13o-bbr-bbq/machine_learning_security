# -*- coding: utf-8 -*-
class Gene:

    genom_list = None
    evaluation = None

    def __init__(self, genom_list, evaluation):
        self.genom_list = genom_list
        self.evaluation = evaluation

    def getGenom(self):
        return self.genom_list

    def getEvaluation(self):
        return self.evaluation

    def setGenom(self, genom_list):
        self.genom_list = genom_list

    def setEvaluation(self, evaluation):
        self.evaluation = evaluation
