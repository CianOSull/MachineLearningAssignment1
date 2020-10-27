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

def task2(training_data):
    # Index stores the current enumerated count in this for loop
    # row value contians the string from training_data
    for index, row_value in enumerate(training_data['Review']):
        # Basically this just goes the entire string and puts all the alphanum
        # and white space characters into a new string.
        # White space characters are kept so the string can be split
        transformedValue = "".join(c for c in row_value if c.isalnum() or c == " ")
        transformedValue = transformedValue.lower()
        transformedValue = transformedValue.split()
        
        # Set the value of the row in training to transformedValue
        # iloc gets a row from the dataframe as a series with the index put in
        # values gets all the values in that seires
        training_data.iloc[index].values[index] = transformedValue
        break
    
    print(training_data.iloc[0].values[0])
    
def task3():
    pass

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
    task2(training_data)
     
     
print(("="*50), "Main", ("="*50))
main()



