#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:19:30 2017

@author: zhenshan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm


def SVMClassification(featureDF, trainProp = 0.6):
    '''SVM with Cross validation;
       return test error'''
    # coding phonies
    X = featureDF.ix[:,1:7]
    Y = pd.DataFrame(featureDF.ix[:,"phone name"].astype('category'))
    cat_columns = Y.select_dtypes(['category']).columns
    Y[cat_columns] = Y[cat_columns].apply(lambda x: x.cat.codes)
    
    # split data into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - trainProp), random_state=0)
    
    # build SVM
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    testError = clf.score(X_test, y_test)    
    
    return testError
