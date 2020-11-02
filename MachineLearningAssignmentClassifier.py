# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:57:32 2020

@author: Cian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# This is task 2
def create_word_list(training_data, min_word_length, min_word_occ):
    word_list = []
    word_occurences = {}
    
    for index, row_value in enumerate(training_data['Review']):
        transformedValue = "".join(c for c in row_value if c.isalnum() or c == " ")
        transformedValue = transformedValue.lower()
        transformedValue = transformedValue.split()
        
        for word in transformedValue:
            if (min_word_length <= len(word)):
                if (word in word_occurences):
                    word_occurences[word] = word_occurences[word] + 1
                else:
                    word_occurences[word]=1    
    
    for word in word_occurences:
        if min_word_occ <= word_occurences[word]:
            word_list.append(word)
            
    print("Word List size: ", len(word_list))
    
    print(("="*50))

    return word_list

def count_word_per_reviews(word_list, training_data, training_labels):        
    positive_word_reivew_count = dict.fromkeys(word_list, 0)
    negative_word_reivew_count = dict.fromkeys(word_list, 0)
    
    for index, row_value in enumerate(training_data['Review']):
        transformedValue = "".join(c for c in row_value if c.isalnum() or c == " ")
        transformedValue = transformedValue.lower()
        transformedValue = transformedValue.split()
               
        for word in word_list:
            if word in transformedValue:
                # Here check if the reivew is ngegative or positive
                if training_labels.iloc[index].values[0] == "positive": 
                   positive_word_reivew_count[word] = positive_word_reivew_count[word] + 1
                   continue
                else:
                   negative_word_reivew_count[word] = negative_word_reivew_count[word] + 1
                   continue
                    
    
    print("Task 3 done")
    print(("="*50))
    
    return positive_word_reivew_count, negative_word_reivew_count

def calc_liklihoods(positive_word_reivew_count, negative_word_reivew_count, total_positive, total_negative):
    # The total reviews of course would be just hte two added together
    total_reviews = total_positive + total_negative
    
    alpha = 1
    
    likelihood_positive = positive_word_reivew_count.copy()
    for word in positive_word_reivew_count:
        likelihood_positive[word] = (positive_word_reivew_count[word] + alpha)/(total_positive + 2*alpha)
    
    likelihood_negative = positive_word_reivew_count.copy()
    for word in negative_word_reivew_count:
        likelihood_negative[word] = (negative_word_reivew_count[word] + alpha)/(total_negative + 2*alpha)
     
    print("Likelihood positive word: ", list(positive_word_reivew_count)[0], "Ratio:", likelihood_positive[list(positive_word_reivew_count)[0]])
    print("Negative positive word: ", list(negative_word_reivew_count)[0], "Ratio:", likelihood_negative[list(negative_word_reivew_count)[0]])
    
    prior_review_pos = total_positive/total_reviews
    prior_review_neg = total_negative/total_reviews
    
    print("Prior Positive Ratio: ", prior_review_pos)
    print("Prior Negative Ratio: ", prior_review_neg)
    
    print("Task 4 done")
    print(("="*50))
    
    return likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg

def classifier(review_text, likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg):
    logLikelihood_positive = 0
    logLikelihood_negative = 0
    
    transformedValue = "".join(c for c in review_text if c.isalnum() or c == " ")
    transformedValue = transformedValue.lower()
    transformedValue = transformedValue.split()
    
    for word in transformedValue:
        # if itis in liklihood text then it should be in negative as well
        if word in likelihood_positive:
            logLikelihood_positive = logLikelihood_positive + math.log(likelihood_positive[word])
            logLikelihood_negative = logLikelihood_negative + math.log(likelihood_negative[word])
    
    if math.log(prior_review_pos) - math.log(prior_review_neg) < logLikelihood_positive - logLikelihood_negative:
        # This one should be 1 for positive
        prediction = 1
    else:
        # This one should be 0 for negative
        prediction = 0 
            
    return prediction

def main():
    # Do task 6 here
    pass
    
print(("="*50), "Main", ("="*50))
main()

