#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import math
import re


class NaiveBayes:
    def __init__(self):
        self.vocabularies = set()
        self.word_count = {}
        self.category_count = {}
        self.file_name = os.path.basename(__file__)

    # Count up word (Create Bag-of-Words).
    def word_count_up(self, word, category):
        self.word_count.setdefault(category, {})
        self.word_count[category].setdefault(word, 0)
        self.word_count[category][word] += 1
        self.vocabularies.add(word)

    # Count up category number.
    def category_count_up(self, category):
        self.category_count.setdefault(category, 0)
        self.category_count[category] += 1

    # Learning based on keyword and category.
    def train(self, doc, category):
        # Count each category.
        self.word_count_up(doc, category)
        # Count category number.
        self.category_count_up(category)

    # Calculate prior probability of Bayes.
    def prior_prob(self, category):
        num_of_categories = sum(self.category_count.values())
        num_of_docs_of_the_category = self.category_count[category]
        return float(num_of_docs_of_the_category) / float(num_of_categories)

    # Count number of appearance.
    def num_of_appearance(self, word, category):
        word_count = 0
        keyword_list = []
        for key_item in self.word_count[category]:
            list_match = re.findall(key_item, word, flags=re.IGNORECASE)
            if len(list_match) != 0:
                word_count += 1
                for item in list_match:
                    keyword_list.append(item)
        prob = float(word_count) / float(len(self.word_count[category]))
        return word_count, list(set(keyword_list)), prob

    # Calculate Bayes.
    def word_prob(self, word, category):
        numerator, keyword_list, temp_prob = self.num_of_appearance(word, category)
        # Laplace  smoothing.
        numerator += 1
        denominator = sum(self.word_count[category].values()) + len(self.vocabularies)
        prob = float(numerator) / float(denominator)
        return prob, keyword_list, temp_prob

    # Calculate score.
    def score(self, word, category):
        score = math.log(self.prior_prob(category))
        prob, keyword_list, temp_prob = self.word_prob(word, category)
        score += math.log(prob)
        return score, prob, keyword_list, temp_prob

    # Execute classify.
    def classify(self, doc):
        best_guessed_category = None
        max_prob_before = -sys.maxsize
        keyword_list = []
        classified_list = []

        # Calculate score each category.
        for category in self.category_count.keys():
            score, total_prob, feature_list, category_prob = self.score(doc, category)
            classified_list.append([category, float(total_prob), feature_list])

            # Classify word to highest score's category.
            if score > max_prob_before:
                max_prob_before = score
                best_guessed_category = category
                keyword_list = feature_list
                classified_prob = total_prob
        return best_guessed_category, float(classified_prob), keyword_list, classified_list
