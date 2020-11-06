# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:36:27 2020

@author: Cian O'Sullivan
Student number: R00160696
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import model_selection
from sklearn import metrics


def task1():
    # First load the excel sheet and take a peek at whats inside
    review_df = pd.read_excel("movie_reviews.xlsx")
    review_df['Sentiment'] = review_df['Sentiment'].map({'negative' : 0, 'positive' : 1})
    
    # Create the four lists
    # This gets every row in review where split = train and creats a df with 
    # the review column
    training_data = review_df[review_df['Split'] == "train"][['Review']]
    
    training_labels = review_df[review_df['Split'] == "train"][['Sentiment']]
    
    test_data = review_df[review_df['Split'] == "test"][['Review']]
    
    test_labels = review_df[review_df['Split'] == "test"][['Sentiment']]
    
    # # Print the count of labels for training and test
    # print(len(training_labels[training_labels["Sentiment"] == "positive"]))
    
    # Task 4 needs total for prior
    total_reviews = len(training_labels)
    # print(total_reviews)
    # Task 4 mentions that these were obtained in task 1 so thats why they are here to be returned
    total_positive = len(training_labels[training_labels["Sentiment"] == 1])
    total_negative = len(training_labels[training_labels["Sentiment"] == 0])
    
    print("The number of postive reviews in the training set is >>>: ", 
          len(training_labels[training_labels["Sentiment"] == 1]))
    
    print("The number of negative reviews in the training set is >>>: ", 
          len(training_labels[training_labels["Sentiment"] == 0]))
    
    print("The number of postive reviews in the evaluation set is >>>: ", 
          len(test_labels[test_labels["Sentiment"] == 1]))
    
    print("The number of negative reviews in the evaluation set is >>>: ",
          len(test_labels[test_labels["Sentiment"] == 0]))
    print(("="*50))
    
    return training_data, training_labels, test_data, test_labels, total_positive, total_negative, total_reviews

def task2(training_data, min_word_length, min_word_occ):
    word_list = []
    word_occurences = {}
    
    # Index stores the current enumerated count in this for loop
    # row value contians the string from training_data
    for index, row_value in enumerate(training_data['Review']):
        # Basically this just goes the entire string and puts all the alphanum
        # and white space characters into a new string.
        # White space characters are kept so the string can be split
        transformedValue = "".join(c for c in row_value if c.isalnum() or c == " ")
        transformedValue = transformedValue.lower()
        transformedValue = transformedValue.split()
        
        # THIS PART NOT BE NEEDED
        # Set the value of the row in training to transformedValue
        # iloc gets a row from the dataframe as a series with the index put in
        # values gets all the values in that seires
        # training_data.iloc[index].values[0] = transformedValue
        
        # Get the word occurences for each word and add each word to dict
        for word in transformedValue:
            if (min_word_length <= len(word)):
                if (word in word_occurences):
                    word_occurences[word] = word_occurences[word] + 1
                else:
                    word_occurences[word]=1    
    
    # Add each word that occurs min times into word list
    for word in word_occurences:
        if min_word_occ <= word_occurences[word]:
            word_list.append(word)
            
    # Just to check the size of word list
    print(len(word_list))
    
    print(("="*50))

    return word_list
    
def task3(word_list, training_data, training_labels):        
    # This will create a dictionary where each value from word list is the key
    # for the count of the amount of times that word appears.
    # Also dict.fromkeys(list,y) creates a dictionary with a list as a key
    # and whatever base values you want.
    positive_word_reivew_count = dict.fromkeys(word_list, 0)
    negative_word_reivew_count = dict.fromkeys(word_list, 0)
    
    # This checks if the a word appears in all positive and neagtive reviews 
    # Get a list of all the words a list of strings to check
    for index, row_value in enumerate(training_data['Review']):
        transformedValue = "".join(c for c in row_value if c.isalnum() or c == " ")
        transformedValue = transformedValue.lower()
        transformedValue = transformedValue.split()
               
        # For each word in word list
        for word in word_list:
            # Check if the word is in the string list
            if word in transformedValue:
                # Here check if the reivew is ngegative or positive
                # if training_labels.iloc[index].values[0] == "positive": 
                if training_labels.iloc[index].values[0] == 1: 
                   # If it is then increment its value in the dictionary
                   positive_word_reivew_count[word] = positive_word_reivew_count[word] + 1
                   # Then continue unto the next review
                   continue
                else:
                   # If it is then increment its value in the dictionary
                   negative_word_reivew_count[word] = negative_word_reivew_count[word] + 1
                   # Then continue unto the next review
                   continue
                    
    
    print("Task 3 done")
    print(("="*50))
    
    return positive_word_reivew_count, negative_word_reivew_count

def task4(positive_word_reivew_count, negative_word_reivew_count, total_positive, total_negative, total_reviews):
    # Consider each word extracted in task 2 as a binary feature of a review 
    # indicating that a word is either present in the review or absent in the 
    # review. 
    
    
    # Using the function created in task 3 to count the number of reviews each 
    # of these features is present in, calculate the likelihoods
    # 𝑃[𝑤𝑜𝑟𝑑 𝑖𝑠 𝑝𝑟𝑒𝑠𝑒𝑛𝑡 𝑖𝑛 𝑟𝑒𝑣𝑖𝑒𝑤|𝑟𝑒𝑣𝑖𝑒𝑤 𝑖𝑠 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒]
    # and
    # 𝑃[𝑤𝑜𝑟𝑑 𝑖𝑠 𝑝𝑟𝑒𝑠𝑒𝑛𝑡 𝑖𝑛 𝑟𝑒𝑣𝑖𝑒𝑤|𝑟𝑒𝑣𝑖𝑒𝑤 𝑖𝑠 𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒]
    # for each word in the feature vector.
    # Create a function that calculates these likelihoods for all words 
    # applying Laplace smoothing with a smoothing factor 𝛼 = 1 [1 point]. 
    alpha = 1
    
    # Make dictionary with every word as keys and it will have the ratios
    likelihood_positive = positive_word_reivew_count.copy()
    # 𝑃[𝑤𝑜𝑟𝑑 𝑖𝑠 𝑝𝑟𝑒𝑠𝑒𝑛𝑡 𝑖𝑛 𝑟𝑒𝑣𝑖𝑒𝑤|𝑟𝑒𝑣𝑖𝑒𝑤 𝑖𝑠 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒]
    for word in positive_word_reivew_count:
        # so its positive review count for word / total number of positive reivews
        likelihood_positive[word] = (positive_word_reivew_count[word] + alpha)/(total_positive + 2*alpha)
    
    # 𝑃[𝑤𝑜𝑟𝑑 𝑖𝑠 𝑝𝑟𝑒𝑠𝑒𝑛𝑡 𝑖𝑛 𝑟𝑒𝑣𝑖𝑒𝑤|𝑟𝑒𝑣𝑖𝑒𝑤 𝑖𝑠 𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒]
    likelihood_negative = positive_word_reivew_count.copy()
    # 𝑃[𝑤𝑜𝑟𝑑 𝑖𝑠 𝑝𝑟𝑒𝑠𝑒𝑛𝑡 𝑖𝑛 𝑟𝑒𝑣𝑖𝑒𝑤|𝑟𝑒𝑣𝑖𝑒𝑤 𝑖𝑠 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒]
    for word in negative_word_reivew_count:
        # so its positive review count for word / total number of positive reivews
        likelihood_negative[word] = (negative_word_reivew_count[word] + alpha)/(total_negative + 2*alpha)
    
    
    print("Likelihood positive word: ", list(positive_word_reivew_count)[0], "Ratio:", likelihood_positive[list(positive_word_reivew_count)[0]])
    print("Negative positive word: ", list(negative_word_reivew_count)[0], "Ratio:", likelihood_negative[list(negative_word_reivew_count)[0]])
    
    # The function should take the two mappings created in task 3 and the 
    # total number of positive/negative reviews obtained in task 1 as input 
    # and return a dictionary mapping each feature word to the likelihood 
    # probability that a word is present in a review given its class being 
    # either positive or negative.
    
    # Also calculate the priors
    # 𝑃[𝑟𝑒𝑣𝑖𝑒𝑤 𝑖𝑠 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒]
    # and
    # 𝑃[𝑟𝑒𝑣𝑖𝑒𝑤 𝑖𝑠 𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒]
    # by considering the fraction of positive/negative reviews in the training 
    # set [1 point].
    prior_review_pos = total_positive/total_reviews
    prior_review_neg = total_negative/total_reviews
    
    print("Prior Positive Ratio: ", prior_review_pos)
    print("Prior Negative Ratio: ", prior_review_neg)
    
    print("Task 4 done")
    print(("="*50))
    
    return likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg

def task5(likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg, test_data, test_labels, word_list):
    # Use the likelihood functions and priors created in task 4 to now create 
    # a Naïve Bayes classifier for predicting the sentiment label for a new 
    # review text [2 points].
    # Remember to use logarithms of the probabilities for numerical stability. 
    
    prediction = []
    for index, row_value in enumerate(test_data['Review']):
        logLikelihood_positive = 0
        logLikelihood_negative = 0
        
        transformedValue = "".join(c for c in row_value if c.isalnum() or c == " ")
        transformedValue = transformedValue.lower()
        transformedValue = transformedValue.split()
               
        # print(transformedValue)
        
        for word in transformedValue:
            if word in word_list:
                logLikelihood_positive = logLikelihood_positive + math.log(likelihood_positive[word])
                logLikelihood_negative = logLikelihood_negative + math.log(likelihood_negative[word])
        
        # print("Log postive: ", logLikelihood_positive)
        # print("Log negative: ", logLikelihood_negative)
        
        if math.log(prior_review_pos) - math.log(prior_review_neg) < logLikelihood_positive - logLikelihood_negative:
            # This one should be 1 for positive
            prediction.append(1)
        else:
            # This one should be 0 for negative
            prediction.append(0)
    
    # The function should take as input the new review text as string as well 
    # as the priors and likelihoods calculated in task 4. It should produce as 
    # output the predicted sentiment label for the new review 
    # (i.e. either “positive” or “negative”).
            
    print("Task 5 done")
    print(("="*50))
    
    return prediction

def classifier(feature_data, target_labels):
    word_list = []
    word_occurences = {}
    min_word_length = 3
    min_word_occ = 10000
    
    for index in range(len(feature_data)):
        transformedValue = "".join(c for c in feature_data.values[index][0] if c.isalnum() or c == " ")
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
            
    print(word_list)
    print(("="*50))
    # ======================================================
    
    positive_word_reivew_count = dict.fromkeys(word_list, 0)
    negative_word_reivew_count = dict.fromkeys(word_list, 0)
    
    for index in range(len(feature_data)):
        transformedValue = "".join(c for c in feature_data.values[index][0] if c.isalnum() or c == " ")
        transformedValue = transformedValue.lower()
        transformedValue = transformedValue.split()
               
        for word in word_list:
            if word in transformedValue: 
                if target_labels.values[index] == 1: 
                    positive_word_reivew_count[word] = positive_word_reivew_count[word] + 1
                    continue
                else:
                    negative_word_reivew_count[word] = negative_word_reivew_count[word] + 1
                    continue

    
    print(positive_word_reivew_count)
    print("===")
    print(negative_word_reivew_count)
    print(("="*50))
    
def task6():
    # Create a k-fold cross-validation procedure for splitting the training 
    # set into k folds 
    kf = model_selection.KFold(n_splits=6, shuffle=True)
    
    # and train the classifier created in tasks 2-5 on the training subset [1 point]. 
    review_df = pd.read_excel("movie_reviews.xlsx")
    review_df['Sentiment'] = review_df['Sentiment'].map({'negative' : 0, 'positive' : 1})
    
    training_data = review_df[review_df['Split'] == "train"][['Review']]
    
    # training_labels = review_df[review_df['Split'] == "train"][['Sentiment']]
    
    training_labels = review_df['Sentiment']
    
    for train_index, test_index in kf.split(training_data):
        # training_data.iloc[index]
        # print("Train index", train_index) 
        # print("Test index: ", test_index)
        # print("Traning data subset using train_index: ", training_data.iloc[train_index])
        classifier(training_data.iloc[train_index], training_labels[train_index])
        
        # Confusion matrix 
        # c = metrics.confusion_matrix(target[test_index], results)
        
    # Evaluate the classification accuracy, i.e. the fraction of correctly 
    # classifier samples, on the evaluation subset [1 point]
    
    # and use this procedure to calculate the mean accuracy score [1 point].
   
    # Compare different accuracy scores for different choices 
    # (1,2,3,4,5,6,7,8,9,10) of the word length parameter as defined in task 2 [1 point]. 
    
    # Select the optimal word length parameter [1 point] 
    # and evaluate the resulting classifier on the test set extracted in task 1.
    
    # The final evaluation should contain:
    # - The confusion matrix for the classification [1 point]
    # - The percentage of true positive [1 point], true negatives [1 point], false positives [1
    # point] and false negatives [1 point]
    # - The classification accuracy score, i.e. the fraction of correctly classified samples [1
    # point]

    pass

def main():
    # First create the four needed lists 
    # training_data, training_lables, test_data, test_labels, total_positive, total_negative, total_reviews = task1()
    
    # # Now setup training data
    # # Setting a min word count of 4 because manjority of non sentiment words
    # # like "the" or "and" probably arnt needed.  Will skip "but" though
    # # Keep min word occ small though for testing as large count can take a while
    # word_list = task2(training_data, 4, 10000)
    
    # # # Now run task3
    # positive_word_reivew_count, negative_word_reivew_count = task3(word_list, training_data, training_lables)
    
    # # # Now do task4
    # likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg = task4(positive_word_reivew_count, negative_word_reivew_count, total_positive, total_negative, total_reviews)
    
    # # # Now do task5
    # prediction = task5(likelihood_positive, likelihood_negative, prior_review_pos, prior_review_neg, test_data, test_labels, word_list)
    
    task6()
    
print(("="*50), "Main", ("="*50))
main()



