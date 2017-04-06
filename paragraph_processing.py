#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 18:32:35 2017

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
import SentenceSegmentation
import Visualization
import Classification

#==============================================================================
# Data Loading(pandas trunk reading & HDF5)
#==============================================================================
sentenceTrain = load.loadOrderedSentence()# sentence
ecogTrain = load.loadEcog()# ecog data(neural signal)

fileName = 'AlignedTimePara.txt'
rawIntervals = load.loadPhone(fileName) # raw split point data 

timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)# create split point data frame

for sent, pd in timeIntervals.items():
    sentenceFilePath = "data/output/" + sent + ".csv"
    pd.to_csv(sentenceFilePath)