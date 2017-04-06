#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:59:25 2017

@author: zhenshan
"""


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

fileName = 'AlignedTime.txt'
rawIntervals = load.loadPhone(fileName) # raw split point data 

#==============================================================================
# Data Scaling(Parallelized)
#==============================================================================
# Scaling
ecogTrainScaled = DataManipulation.ScalingBySilence(ecogTrain, rawIntervals)

# Scaling Testing
expSentence = 'Doris_ordered_twelve_white_catsAV_RMS'
#All
isTotal = True
result = DataManipulation.ScalingVisualCheck(expSentence, ecogTrain, rawIntervals, isTotal, colIdx = None, plot = False)
#Individual
isTotal = False; plot = True; colIdx = 0
DataManipulation.ScalingVisualCheck(expSentence, ecogTrain, rawIntervals, isTotal, colIdx, plot)

# phone labeling
ecogTrainScaledLabled = DataManipulation.PhoneLabeling(ecogTrainScaled, rawIntervals)

# Output
DataManipulation.ExportScalingData(ecogTrainScaledLabled)

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
    filename = "C:/Users/yangy_000/Dropbox/BAYLOR/TEMP/NeuralSignalDecoding/NODE%d.csv" %i
    featureDF.to_csv(filename)


#### Feature Testing(Only take one node with one frequency for example)
featureList = ['mean']
nodeIdx = [0]
frequency = ['Delta']#, 'Theta', 'Alpha' ,'Beta' ,'Low Gamma', 'High Gamma']
# Generation
featureDF = FeatureGeneration.FeatureDF(ecogTrainScaled, sentenceTrain, nodeIdx, frequency, featureList, rawIntervals)
# Accuracy testing
FeatureGeneration.FeatureVisualCheck(featureDF)

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
