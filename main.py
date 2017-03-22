#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:59:25 2017

@author: zhenshan
"""

'''
@(1)the original data set is fairly large with 2.1GB, here is just very small part of data. 
 Therefore, the visualization and classification performance is not very interesting.
 (2) If the size of dataset is increasing more, distributed computation(Hadoop/Spark) would be the next step

@Total Running time: 10 seconds
'''

# Add function/data folder into module searching path
import sys
import os
sys.path.append(os.getcwd() + '/function')
sys.path.append(os.getcwd() + '/data')

# import Self defined module
import load
import DataManipulation
import FeatureGeneration
import Visualization
import Classification

#==============================================================================
# Data Loading(pandas trunk reading & HDF5)
#==============================================================================
sentenceTrain = load.loadOrderedSentence()# sentence
ecogTrain = load.loadEcog()# ecog data(neural signal)
rawIntervals = load.loadPhone() # raw split point data 


#==============================================================================
# Data Scaling(Parallelized)
#==============================================================================
ecogTrainScaled = DataManipulation.ScalingBySilence(ecogTrain, rawIntervals)


#==============================================================================
# Feature Generation
#==============================================================================
# parameters
#featureList = ['mean'] # take mean as an example
#nodeIdx = [0] # start from 0 to 69(70 in total)
#frequency = ['Delta', 'Theta', 'Alpha' ,'Beta' ,'Low Gamma', 'High Gamma'] # name of frequencies
# Generation
#featureDF = FeatureGeneration.FeatureDF(ecogTrainScaled, sentenceTrain, nodeIdx, frequency, featureList, rawIntervals)

for i in range(0,70):
    featureList = ['mean']
    nodeIdx = [i] # start from 0 to 69(70 in total)
    frequency = ['Delta', 'Theta', 'Alpha' ,'Beta' ,'Low Gamma', 'High Gamma']# ['Delta', 'Theta', 'Alpha' ,'Beta' ,'Low Gamma', 'High Gamma']
    featureDF = FeatureGeneration.FeatureDF(ecogTrainScaled, sentenceTrain, nodeIdx, frequency, featureList, rawIntervals)
    filename = "NODE%d.csv" %i
    featureDF.to_csv(filename)

#==============================================================================
# Data Visualization (Visualized by Dimension reduciton)
#==============================================================================
# parameters
n_neighbors = 10
n_components = 2 # number of reduced dimensions
# t-SNE method(dimension reduction techniques)
Visualization.DimensionReduction(n_neighbors, n_components, featureDF)


#==============================================================================
# Classification (Take SVM as an example)
#==============================================================================
trainProp = 0.6 # training datasize
Classification.SVMClassification(featureDF, trainProp)
