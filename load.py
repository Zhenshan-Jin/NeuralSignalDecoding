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


def loadPhone(fileName):
    '''Reading the raw phone data for each sentence from .txt'''
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)

    # read data
    try:
        rawData_ = open(fileName).read()
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

def loadOrderedSentenceNew():
    '''Reading 210 sentences from .txt'''
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data/NewData'
    os.chdir(dataDir)
    
    # read data
    try:
        sentence = open('Sentence.txt',"r").readlines()
        os.chdir(owd) # switch back to base directory
    except IOError:
        os.chdir(owd)
    
    return sentence


def loadEcog():
    '''Function: Create Dictionary for each sentence with corresponding phone'''
    '''Method: Reading large dataset by pandas chunck and store in HDF5'''
    sentences = loadOrderedSentence()
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)
    
    # read data
    breakPoint = [int(index) for index in open('train_breakpoint.txt').readlines()]
    breakPoint.insert(0,0)
    try:
#        timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)# create split point data frame
        #Read data from HDF5 database
        h5f_read = h5py.File('ecog_training.h5','r')
        train_X_ecog = np.array(h5f_read.get('train_data'))
        train_X_ecog = pd.DataFrame(train_X_ecog)
        
        os.chdir(owd) # switch back to base directory
    except IOError:
        print('create a new HDF5 database for training data')
        try:
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
        except IOError:
            print('Unknown error')
            os.chdir(owd)
    
    train_X_ecog = pd.DataFrame(train_X_ecog)
    ecogData = {}
    for idx in range(len(breakPoint) - 1):
        sentence = util.SentenceAdjustment(sentences[idx]) # helper function: adjust the sentence format to be readable
        ecogData[sentence] = train_X_ecog[breakPoint[idx] : breakPoint[idx + 1]]
        
    return ecogData


def loadEcogNew():
    '''Function: Create Dictionary for each sentence(210) with corresponding phone'''
    '''Method: Reading large dataset by pandas chunck and store in HDF5'''
    sentences = loadOrderedSentenceNew()
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data/NewData'
    os.chdir(dataDir)
    
    # read data
    breakPoint = [int(index) for index in open('breakpoint.txt').readlines()]
    breakPoint.insert(0,0)
    
    breakPointNonSpeech = [int(index) for index in open('silenceBreakpoint.txt').readlines()]
    breakPointNonSpeech.insert(0,0)
    
    try:
#        timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)# create split point data frame
        #Read data from HDF5 database
        h5f_read = h5py.File('ecog.h5','r')
        X_ecog = np.array(h5f_read.get('data'))
        X_ecog_sil = np.array(h5f_read.get('dataNonspeech'))
        h5f_read.close()

        os.chdir(owd) # switch back to base directory
    except IOError:
        print('create a new HDF5 database for training data')
        try:
            # Reading large dataset by chunck and store in HDF5
            h5f = h5py.File('ecog.h5', 'w')
            
            chunksize = 567 # dataset size = 65205; chunk number = 115
            dims = 70 # variable number in dataset
            reader = pd.read_csv('X_ecog.csv', chunksize=chunksize, header=None)
            X_ecog = np.empty((0,dims)).astype(np.float16)
            for chunk in reader:
                # saving memory by customerized data type
                d = np.asarray(chunk.ix[:,:]).astype(np.float16)
                X_ecog = np.vstack((X_ecog, d))
            
            chunksizeNonSpeech = 315 # dataset size = 65205; chunk number = 115
            dimsNonSpeech = 70 # variable number in dataset
            readerNonSpeech = pd.read_csv('X_ecog_sil.csv', chunksize=chunksizeNonSpeech, header=None)
            X_ecog_sil = np.empty((0,dimsNonSpeech)).astype(np.float16)
            for chunkNonSpeech in readerNonSpeech:
                # saving memory by customerized data type
                dNonSpeech = np.asarray(chunkNonSpeech.ix[:,:]).astype(np.float16)
                X_ecog_sil = np.vstack((X_ecog_sil, dNonSpeech))
                
            h5f.create_dataset('data', data=X_ecog, compression="gzip")
            h5f.create_dataset('dataNonspeech', data=X_ecog_sil, compression="gzip")
            h5f.close()
    
            os.chdir(owd) 
        except IOError:
            print('Unknown error')
            os.chdir(owd)
    
    X_ecog = pd.DataFrame(X_ecog)
    X_ecog_sil = pd.DataFrame(X_ecog_sil)

    ecogData = {}
    for idx in range(len(breakPoint) - 1):
        sentence = util.SentenceAdjustment(sentences[idx]) # helper function: adjust the sentence format to be readable
        speechData = X_ecog[breakPoint[idx] : breakPoint[idx + 1]]
        nonSpeechData = X_ecog_sil[breakPointNonSpeech[idx] : breakPointNonSpeech[idx + 1]]
        totalData = pd.concat([nonSpeechData,speechData], axis = 0).reset_index().ix[:,1:]
        ecogData[sentence] = totalData
        
    return ecogData



def loadEcogSilence():
    '''Function: Create Dictionary for each sentence(210) with corresponding phone'''
    '''Method: Reading large dataset by pandas chunck and store in HDF5'''
    sentences = loadOrderedSentenceNew()
    # switch to data directory
    owd = os.getcwd()
    dataDir = owd + '/data/NewData'
    os.chdir(dataDir)
    
    # read data
    breakPoint = [int(index) for index in open('silenceBreakpoint.txt').readlines()]
    breakPoint.insert(0,0)
    
    try:
#        timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)# create split point data frame
        #Read data from HDF5 database
        h5f_read = h5py.File('ecog_sil.h5','r')
        X_ecog_sil = np.array(h5f_read.get('data'))
        X_ecog_sil = pd.DataFrame(X_ecog_sil)
        
        os.chdir(owd) # switch back to base directory
    except IOError:
        print('create a new HDF5 database for training data')
        try:
            # Reading large dataset by chunck and store in HDF5
            h5f = h5py.File('ecog_sil.h5', 'w')
            
            chunksize = 315 # dataset size = 65205; chunk number = 115
            dims = 70 # variable number in dataset
            reader = pd.read_csv('X_ecog_sil.csv', chunksize=chunksize, header=None)
            X_ecog_sil = np.empty((0,dims)).astype(np.float16)
            for chunk in reader:
                # saving memory by customerized data type
                d = np.asarray(chunk.ix[:,:]).astype(np.float16)
                X_ecog_sil = np.vstack((X_ecog_sil, d))

            h5f.create_dataset('data', data=X_ecog_sil, compression="gzip")
            h5f.close()
    
            os.chdir(owd) 
        except IOError:
            print('Unknown error')
            os.chdir(owd)
    
    X_ecog_sil = pd.DataFrame(X_ecog_sil)
    ecogDataSil = {}
    for idx in range(len(breakPoint) - 1):
        sentence = util.SentenceAdjustment(sentences[idx]) # helper function: adjust the sentence format to be readable
        ecogDataSil[sentence] = X_ecog_sil[breakPoint[idx] : breakPoint[idx + 1]]
        
    return ecogDataSil