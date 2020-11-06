# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:57:32 2020

@author: Cian
"""

import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from sklearn import metrics

def process_text(text):
    text = "".join(c for c in text if c.isalnum() or c == " ")
    text = text.lower()
    text = text.split()
    return text

def classifier(feature_data, target_labels, total_positive, total_negative, total_reviews, min_word_length, min_word_occ):
    word_list = []
    word_occurences = {}
    
    for index in range(len(feature_data)):
        transformedValue = process_text(feature_data.values[index][0])
        
        for word in transformedValue:
            if (min_word_length <= len(word)):
                if (word in word_occurences):
                    word_occurences[word] = word_occurences[word] + 1
                else:
                    word_occurences[word]=1    
    
    for word in word_occurences:
        if min_word_occ <= word_occurences[word]:
            word_list.append(word)
            
    positive_word_reivew_count = dict.fromkeys(word_list, 0)
    negative_word_reivew_count = dict.fromkeys(word_list, 0)
    
    for index in range(len(feature_data)):
        transformedValue = process_text(feature_data.values[index][0])
               
        for word in word_list:
            if word in transformedValue: 
                if target_labels.values[index] == 1: 
                    positive_word_reivew_count[word] = positive_word_reivew_count[word] + 1
                    continue
                else:
                    negative_word_reivew_count[word] = negative_word_reivew_count[word] + 1
                    continue

    alpha = 1
    
    likelihood_positive = positive_word_reivew_count.copy()
    
    for word in positive_word_reivew_count:
        likelihood_positive[word] = (positive_word_reivew_count[word] + alpha)/(total_positive + 2*alpha)
    
    likelihood_negative = positive_word_reivew_count.copy()
    for word in negative_word_reivew_count:
        likelihood_negative[word] = (negative_word_reivew_count[word] + alpha)/(total_negative + 2*alpha)
    
    prior_review_pos = total_positive/total_reviews
    prior_review_neg = total_negative/total_reviews
    
    prediction = []
    for index in range(len(feature_data)):
        logLikelihood_positive = 0
        logLikelihood_negative = 0
        
        transformedValue = process_text(feature_data.values[index][0])
        
        for word in transformedValue:
            if word in word_list:
                logLikelihood_positive = logLikelihood_positive + math.log(likelihood_positive[word])
                logLikelihood_negative = logLikelihood_negative + math.log(likelihood_negative[word])
        
        if math.log(prior_review_pos) - math.log(prior_review_neg) < logLikelihood_positive - logLikelihood_negative:
            prediction.append(1)
        else:
            prediction.append(0) 
    
    return prediction
    
def task6():
    kf = model_selection.KFold(n_splits=6, shuffle=True)
    
    training_data, training_lables, test_data, test_labels, total_positive, total_negative, total_reviews = task1()
    
    all_results = []
    mean_results = []
    
    for i in range(1,11):
        for train_index, test_index in kf.split(training_data):
            results = classifier(training_data.iloc[train_index], training_lables.iloc[train_index], total_positive, total_negative, total_reviews, i, 10000)
            
            all_results.append(metrics.accuracy_score(results, training_lables.iloc[train_index]))
            print(("="*50))
    
        mean_results.append(np.mean(all_results))
    
    print("Mean Results: ", mean_results)
    highest_min_word_len = mean_results.index(max(mean_results))+1
    print("highest accuracy Min word length: ", highest_min_word_len)
    
    true_positive = []
    true_negative = []
    false_postiive = []
    false_negatives = []
    
    test_results = classifier(test_data, test_labels, total_positive, total_negative, total_reviews, highest_min_word_len, 10000)
    
    all_results.append(metrics.accuracy_score(test_results, test_labels))
    
    C = metrics.confusion_matrix(test_labels, test_results)
    
    true_positive.append(C[0,0])
    true_negative.append(C[1,1])            
    false_postiive.append(C[1,0])
    false_negatives.append(C[0,1])
    
    print(("="*50))

    print("True positives:", round(np.sum(true_positive)/len(test_data), 5), "%")
    print("True negatives:", round(np.sum(true_negative)/len(test_data), 5), "%")
    print("False positives:", round(np.sum(false_postiive)/len(test_data), 5), "%")
    print("False negatives:", round(np.sum(false_negatives)/len(test_data), 5), "%")
            
    print("Test Accuracy: ", np.mean(all_results))

def main():
    task6()
    
print(("="*50), "Main", ("="*50))
main()

