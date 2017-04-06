#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:07:52 2017

@author: zhenshan
"""

import pandas as pd
import numpy as np
import pickle # Export python data 


def AlignmentPoint(rawData_):
    '''Extract sentence/word/phone time intervals from rawIntervals
       {Key1: sentence, Value1: phone time interval data frame}'''
    try:
        sentence = pickle.load( open( "data/Align.p", "rb" ) )
    except IOError:
        dataTrunc = [x.strip() for x in rawData_.split('"/tmp/tmpvzeiz6_9/audio/')]
        del dataTrunc[0]# Remove the first row
        
        sentence = {}# dictionary with key: sentence & value: phone time interval
        for rawSentData in dataTrunc:
            sentLineSplit = rawSentData.splitlines()
            del sentLineSplit[-1]# Remove the last empty line
            subSentence = sentLineSplit[0][:-5]# extract the sentence name
            
            # Extract time interval for each phone in a structured way
            rawTimeInterval = sentLineSplit[1:]
            timeInterval = pd.DataFrame(columns=['start time', 'end time', 'phone', 'word'])
            timeInvervalWord = None
            for interIdx in range(len(rawTimeInterval)):
                timeInvervalSplited = rawTimeInterval[interIdx].split()
                if len(timeInvervalSplited) == 4: # record the word for each phone
                    timeInvervalWord = timeInvervalSplited[3]
                if timeInvervalSplited[2] == 'sp': # mingle the 'sp'(seperate) time to words
                    previous = timeInterval.loc[(interIdx - 1)]
                    previous[1] = timeInvervalSplited[1]
                    timeInterval.loc[(interIdx - 1)] = previous
                    timeInvervalSplited[0] = timeInvervalSplited[1]
                timeInterval.loc[interIdx] = timeInvervalSplited + [timeInvervalWord] if len(timeInvervalSplited) < 4 else timeInvervalSplited
            sentence[subSentence] = timeInterval
            
        pickle.dump(sentence, open( "data/Align.p", "wb" ))
        
    return sentence


def SplitPoint(rawTimeInterval_, neuralSignalLen):
    '''generate the splitting points for each phone in the sentence by proportion'''
    rawTimeInterval_.ix[:,[0,1]] = rawTimeInterval_.ix[:,[0,1]].apply(pd.to_numeric) # transform string to number
    point = rawTimeInterval_.ix[:,0]/100000.0
    totalLength = int(rawTimeInterval_.ix[rawTimeInterval_.shape[0] - 1,1])/100000.0
    point = point.append(pd.Series([totalLength]))
    proportion = np.cumsum(point.diff(1)[1:]/totalLength)
    phoneTimes = [round(x) for x in list(proportion * neuralSignalLen)]
    phoneTimes.insert(0, 0)
    
    return phoneTimes


def NeuralSignalSegmentation(phoneTimes_, neuralSignal_, timeInterval_):
    '''Generate series for phones in each word'''
    phoneSegmentation_ = {} # dict{word: [{phone1: signalData}, {phone2: signalData},......]}
    for phoneIdx in range(timeInterval_.shape[0]):
        signalData = neuralSignal_[(phoneTimes_[phoneIdx]):(phoneTimes_[phoneIdx + 1])]
        phone = timeInterval_.ix[phoneIdx, 2]
        word = timeInterval_.ix[phoneIdx, 3]
        phoneDataDict = {"phone":phone, "signalData": signalData}
        if word in phoneSegmentation_:
            # if key exist, append the value to list
            phoneSegmentation_[word].append(phoneDataDict)
        else:
            # if key not eixist, create a list
            phoneSegmentation_[word] = [phoneDataDict]

    return phoneSegmentation_