# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:36:27 2020

@author: Cian O'Sullivan
Student number: R00160696
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task1():
    # First load the excel sheet and take a peek at whats inside
    review_df = pd.read_excel("movie_reviews.xlsx")
    # print(review_df.head())
    # print(review_df.columns.values)
    # print("="*50)
    
    # Create the four lists
    # This gets every row in review where split = train
    training_data = review_df[review_df['Split'] == "train"][['Review']]
    
    training_labels = review_df[review_df['Split'] == "train"][['Sentiment']]
    
    test_data = review_df[review_df['Split'] == "test"][['Review']]
    
    test_labels = review_df[review_df['Split'] == "train"][['Sentiment']]
    
    # Print the count of labels for training and test
    print(len(training_labels[training_labels["Sentiment"] == "positive"]))
    
    print("The number of postive reviews in the training set is >>>: ", 
          len(training_labels[training_labels["Sentiment"] == "positive"]))
    
    print("The number of negative reviews in the training set is >>>: ", 
          len(training_labels[training_labels["Sentiment"] == "negative"]))
    
    print("The number of postive reviews in the evaluation set is >>>: ", 
          len(training_labels[test_labels["Sentiment"] == "positive"]))
    
    print("The number of negative reviews in the evaluation set is >>>: ",
          len(training_labels[test_labels["Sentiment"] == "negative"]))
    print(("="*50))
    
    return training_data, training_labels, test_data, test_labels

def task2(training_data, min_word_length, min_word_occ):
    world_list = []
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
        training_data.iloc[index].values[0] = transformedValue
        
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
            world_list.append(word)

    return world_list
    
def task3(word_list, training_data, training_labels):
    # Task requirements broken down    
    # Now create a function that goes through all positive reviews in the 
    # training set and counts for each of these words the number of reviews 
    # the word appears in [1 point]. 
    # Do the same for all negative reviews as well [1 point].
    
    # This will create a dictionary where each value from word list is the key
    # for the count of the amount of times that word appears.
    # Also dict.fromkeys(list,y) creates a dictionary with a list as a key
    # and whatever base values you want.
    postive_word_count = dict.fromkeys(word_list, 0)
    
    # Go through each review
    for index, row_value in enumerate(training_data['Review']):
        # if the index current value of the training label is positive
        if training_labels.iloc[index].values[0] == "positive":
            # check each string in training data index
            for string in training_data.iloc[index].values[0]:
                # Get each word from word list
                for word in word_list:
                    # if word is equal to the string
                    if word == string:
                        postive_word_count[word] += 1

def task4():
    pass

def task5():
    pass

def task6():
    pass

def task7():
    pass

def main():
    # First create the four needed lists 
    training_data, training_lables, test_data, test_labels = task1()
    
    # Now setup training data
    word_list = task2(training_data, 3, 300)
    
    # Now run task3
    task3(word_list, training_data, training_lables)
    
    
     
print(("="*50), "Main", ("="*50))
main()



