#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:29:08 2021

@author: matthewayala
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# Simulate logistic data with/without noise

def logistic(x, b, noise=None):
    L = x.T.dot(b)
    if noise is not None:
        L = L+noise
    return 1/(1+np.exp(-L))

# Model fitting parameters used in get_noise_accuracy
MAX_ITER = 1000
N_SPLITS = 5
SEED = 1
# Function that simulates logistic data, trains and validates model on the data.
def get_noise_accuracy(noise_list,x_values,cut_off):
    """Returns the noise iterable, the list of accuracy scores, and the list of standard deviations for error bars."""
    bias = np.ones(len(x_values))
    X = np.vstack([x_values,bias]) # Add Intercept
    B =  [1., 1.] # Sigmoid params for X   
    
    mean_acc_scores = []
    standard_deviations = []

    for i in noise_list:
        # Creating simulated logistic data set with several noise values
        pnoisy = logistic(X, B, noise=np.random.normal(loc=0., scale= i, size=len(x_values)))
        
        # Reshaping data for model    
        x_data = x_values.reshape(len(x_values), 1)
        pnoisy_data = pnoisy.reshape(len(x_values), 1)
        transformed_p = [] # Empty list to hold new transformed data (1's and 0's)
        for a in pnoisy_data:    
            if a >= cut_off: # Turning values > 0.5 in 1's and < 0.5 into 0's
                a = 1
            else:
                a = 0
            transformed_p.append(a) # Adding transformed values to the empty list.
        
        # Turning labels into an array for stratifiedkfold
        transformed_p = np.array(transformed_p)
        # Creating Logistic regression instance
        LogReg = LogisticRegression(max_iter= MAX_ITER)
        
        # StratifiedKFold Splits
        skf = StratifiedKFold(n_splits = N_SPLITS, random_state= SEED, shuffle = True)
        # kf = KFold(n_splits = 5)
        accuracy_scores = [] # Empty accuracy score list to be filled with each iteration of kfold
        for train_index, test_index in skf.split(x_data,transformed_p):
            X_train, X_test = x_data[train_index], x_data[test_index]
            y_train, y_test = transformed_p[train_index], transformed_p[test_index]
            LogReg.fit(X_train,y_train)
            score = LogReg.score(X_test,y_test)
            accuracy_scores.append(score)
        
        # Getting average of each k-fold section
        mean_score = np.average(accuracy_scores)
        mean_acc_scores.append(mean_score)
        
        # Getting standard deviation of each k-fold section
        standard_deviation = np.std(accuracy_scores)
        standard_deviations.append(standard_deviation)
        
    return noise_list, mean_acc_scores, standard_deviations


# Creating a list that will be used for the different cutoffs in labeling.
cut_off_list = [0.01, 0.25, 0.5, 0.75, 0.99]


# Looping through cutoffs and graphing each model at each labeling breakpoint
for i in cut_off_list:
    noise, accuracy, errorbars = get_noise_accuracy(np.arange(0,100,5), np.arange(-10,10,0.01), i)    
    plt.errorbar(noise, accuracy, yerr = errorbars, label = str(i))

plt.title('Noise vs. Accuracy')
plt.xlabel('Noise')
plt.ylabel('Accuracy')
plt.legend(title = 'Label Cut-off')


