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

# Processes Text inputed into it for tasks 2 to 4. E.g. gets rid of all non alphanumerics
def process_text(text):
    # This gets rid of all alphanumerics
    text = "".join(c for c in text if c.isalnum() or c == " ")
    text = text.lower()
    text = text.split()
    return text

# Task 1
def load_data():
    # Load dataset
    review_df = pd.read_excel("movie_reviews.xlsx")
    # print(review_df.head())
    
    # Map all values in sentiment to 0 and 1
    review_df['Sentiment'] = review_df['Sentiment'].map({'negative' : 0, 'positive' : 1})
    
    # Split the dataset into train and test
    training_data = review_df[review_df['Split'] == "train"][['Review']]
    
    training_labels = review_df[review_df['Split'] == "train"][['Sentiment']]
    
    test_data = review_df[review_df['Split'] == "test"][['Review']]
    
    test_labels = review_df[review_df['Split'] == "test"][['Sentiment']]
    
    # The details that are to be printed to the console
    print("The number of positive reviews in the training set is >>>: ", 
          len(training_labels[training_labels["Sentiment"] == 1]))
    
    print("The number of negative reviews in the training set is >>>: ", 
          len(training_labels[training_labels["Sentiment"] == 0]))
    
    print("The number of positive reviews in the evaluation set is >>>: ", 
          len(test_labels[test_labels["Sentiment"] == 1]))
    
    print("The number of negative reviews in the evaluation set is >>>: ",
          len(test_labels[test_labels["Sentiment"] == 0]))
    
    print(("="*50))
    
    return training_data, training_labels, test_data, test_labels

# This does tasks 2 - 4
def create_classifier(feature_data, target_labels, min_word_length, min_word_occ):
    # ============================ Task 2 ============================
    word_list = []
    word_occurences = {}
    
    for index in range(len(feature_data)):
        # Process the text
        transformedValue = process_text(feature_data.values[index][0])
        
        for word in transformedValue:
            if (min_word_length <= len(word)):
                if (word in word_occurences):
                    word_occurences[word] = word_occurences[word] + 1
                else:
                    word_occurences[word]=1    
    
    # Add the words that meet the requirments to hte word list
    for word in word_occurences:
        if min_word_occ <= word_occurences[word]:
            word_list.append(word)
            
    # print(word_list)
    # print(("="*50))
    # ================================================================
    
    # ============================ Task 3 ============================    
    # This will create a dictionary where each value from word list is the key
    # for the count of the amount of times that word appears.
    # Also dict.fromkeys(list,y) creates a dictionary with a list as a key
    # and whatever base values you want.
    positive_word_reivew_count = dict.fromkeys(word_list, 0)
    negative_word_reivew_count = dict.fromkeys(word_list, 0)
    
    for index in range(len(feature_data)):
        transformedValue = process_text(feature_data.values[index][0])
        
        # For each word in word list, if word is in the review, check
        # if that reviews correspond index in target_labels sentiment
        # to choose which dictionary to put it in
        for word in word_list:
            if word in transformedValue: 
                if target_labels.values[index] == 1: 
                    positive_word_reivew_count[word] = positive_word_reivew_count[word] + 1
                    continue
                else:
                    negative_word_reivew_count[word] = negative_word_reivew_count[word] + 1
                    continue
    
    # print(positive_word_reivew_count)
    # print("===")
    # print(negative_word_reivew_count)
    # print(("="*50))
    # ================================================================
    
    # ============================ Task 4 ============================
    # Alpha for laplace smoothing
    alpha = 1

    # Calce the totals
    total_reviews = len(feature_data)
    total_positive = len(target_labels[target_labels["Sentiment"] == 1])
    total_negative = len(target_labels[target_labels["Sentiment"] == 0])
    # print(total_reviews)
    # print(total_positive)
    # print(total_negative)
    
    # Create the two lilehood dictionaries to be used and set all their values to default to 0
    likelihood_positive = dict.fromkeys(word_list, 0)
    likelihood_negative = dict.fromkeys(word_list, 0)
    
    for word in positive_word_reivew_count:
        likelihood_positive[word] = (positive_word_reivew_count[word] + alpha)/(total_positive + 2*alpha)
    
    for word in negative_word_reivew_count:
        likelihood_negative[word] = (negative_word_reivew_count[word] + alpha)/(total_negative + 2*alpha)
    
    # Create the priors
    prior_review_pos = total_positive/total_reviews
    prior_review_neg = total_negative/total_reviews
    
    # print(likelihood_negative)
    # print("===")
    # print(likelihood_negative)
    # print(("="*50))
    
    return likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg
    # ================================================================

# This does task 5
def classifier(feature_data, likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg):
    # ============================ Task 5 ============================
    prediction = []
    for index in range(len(feature_data)):
        logLikelihood_positive = 0
        logLikelihood_negative = 0
        
        transformedValue = process_text(feature_data.values[index][0])
        
        for word in transformedValue:
            # Every key of the likihood dictionary is a word from the word list
            # So if one of the words from the review is in the dictionary:
            if word in likelihood_positive:
                logLikelihood_positive = logLikelihood_positive + math.log(likelihood_positive[word])
                logLikelihood_negative = logLikelihood_negative + math.log(likelihood_negative[word])
        
        if math.log(prior_review_pos) - math.log(prior_review_neg) < logLikelihood_positive - logLikelihood_negative:
            prediction.append(1)
        else:
            prediction.append(0) 
    
    # print(prediction)
    # print(("="*50))
    
    return prediction

def main():
    # ============================ Task 6 ============================
    # Set up the kf split using 6 splits
    kf = model_selection.KFold(n_splits=6, shuffle=True)
    
    # Get the data from the excel file
    training_data, training_lables, test_data, test_labels = load_data()
    
    # These lists are for the accuracy scores
    
    mean_results = []
    
    # For loop for testing 1 - 10 minimum word lengths
    for i in range(1,11):
        all_results = []
        print("Current Min Word Length:", i)
        for train_index, test_index in kf.split(training_data):            
            likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg = create_classifier(training_data.iloc[train_index], training_lables.iloc[train_index], i, 1000)
            results = classifier(training_data.iloc[test_index], likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg)
            
            all_results.append(metrics.accuracy_score(results, training_lables.iloc[test_index]))
        
        print(("="*50))
    
        mean_results.append(np.mean(all_results))
    
    # Prin the accuracy scores
    print("Mean kfold Test Indexes Result Accuracies:", mean_results)
    # Print the mininum word length with the highest accuracy
    # Throuhg multiple tests, at mininum word occurance = 10000, word length of 5 is found to be best
    highest_min_word_len = mean_results.index(max(mean_results))+1
    print("Highest accuracy Min word length:", highest_min_word_len)
    
    # lists for the confusion matrix
    true_positive = []
    true_negative = []
    false_postiive = []
    false_negatives = []

    # Train the classifier again using the word length with the highest accuracy
    likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg = create_classifier(training_data, training_lables, highest_min_word_len, 1000)
    # Now test it on the test data
    test_results = classifier(test_data, likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg)
    
    # This ist he test accuracy
    # This one will contain the accuracy for test
    test_accuracy = metrics.accuracy_score(test_results, test_labels)
    
    C = metrics.confusion_matrix(test_labels, test_results)
    
    true_positive.append(C[0,0])
    true_negative.append(C[1,1])            
    false_postiive.append(C[1,0])
    false_negatives.append(C[0,1])
    
    print(("="*50))
    
    print("True positives:", round(np.sum(true_positive)/len(test_labels), 5), "%")
    print("True negatives:", round(np.sum(true_negative)/len(test_labels), 5), "%")
    print("False positives:", round(np.sum(false_postiive)/len(test_labels), 5), "%")
    print("False negatives:", round(np.sum(false_negatives)/len(test_labels), 5), "%")
    # The accuracy on the test set from task1
    print("Test Accuracy: ", test_accuracy)
    
print(("="*50), "Main", ("="*50))
main()

