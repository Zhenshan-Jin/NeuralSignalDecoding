#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:48:44 2017

@author: zhenshan
"""

import os
import pandas as pd
import numpy as np
import h5py
import util # user-defined module

def loadPhone():
    '''Reading the raw phone data for each sentence from .txt'''
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)

    # read data
    try:
        rawData_ = open('AlignedTime.txt').read()
        os.chdir(owd) # switch back to base directory
    except IOError:
        os.chdir(owd)
    
    return rawData_

def loadOrderedSentence():
    '''Reading sentences from .txt'''
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)
    
    # read data
    try:
        sentence = open('TRAIN.txt',"r").readlines()
        os.chdir(owd) # switch back to base directory
    except IOError:
        os.chdir(owd)

    return sentence

def loadEcog():
    '''Function: Create Dictionary for each sentence with corresponding phone'''
    '''Method: Reading large dataset by pandas chunck and store in HDF5'''
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)
    
    # read data
    try:timeIntervals = FeatureGeneration.AlignmentPoint(rawIntervals)# create split point data frame
        #Read data from HDF5 database
        h5f_read = h5py.File('ecog_training.h5','r')
        train_X_ecog = np.array(h5f_read.get('train_data'))
        train_X_ecog = pd.DataFrame(train_X_ecog)
        
        os.chdir(owd) # switch back to base directory
    except IOError:
        print('create a new HDF5 database')
        try:
            breakPoint = [int(index) for index in open('train_breakpoint.txt').readlines()]
            breakPoint.insert(0,0)
            
            # Reading large dataset by chunck and store in HDF5
            h5f = h5py.File('ecog_training.h5', 'w')
            
            chunksize = 791 # dataset size = 1582; chunk number = 2
            dims = 420 # variable number in dataset
            training_reader = pd.read_csv('train_X_ecog.csv', chunksize=chunksize, header=None)
            train_X_ecog = np.empty((0,dims)).astype(np.float16)
            for chunk in training_reader:
                # saving memory by customerized data type
                d = np.asarray(chunk.ix[:,:]).astype(np.float16)
                train_X_ecog = np.vstack((train_X_ecog, d))

            h5f.create_dataset('train_data', data=train_X_ecog, compression="gzip")
            h5f.close()
    
            os.chdir(owd) 
            sentences = loadOrderedSentence()
        except IOError:
            print('Unknown error')
            os.chdir(owd)

    ecogData = {}
    for idx in range(len(breakPoint) - 1):
        sentence = util.SentenceAdjustment(sentences[idx]) # helper function: adjust the sentence format to be readable
        ecogData[sentence] = train_X_ecog[breakPoint[idx] : breakPoint[idx + 1]]
        
    return ecogData