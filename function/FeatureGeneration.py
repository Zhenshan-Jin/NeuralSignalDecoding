#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:11:06 2017

@author: zhenshan
"""
import pandas as pd
import numpy as np
import unitFeature
import util
import pickle # Export python data 


def AlignmentPoint(rawData_):
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
            if len(timeInvervalSplited) == 4:
                timeInvervalWord = timeInvervalSplited[3]
            timeInterval.loc[interIdx] = timeInvervalSplited + [timeInvervalWord] if len(timeInvervalSplited) < 4 else timeInvervalSplited
        sentence[subSentence] = timeInterval
                
    return sentence


def SplitPoint(rawTimeInterval_, neuralSignalLen):
    rawTimeInterval_.ix[:,[0,1]] = rawTimeInterval_.ix[:,[0,1]].apply(pd.to_numeric) # transform string to number
    point = rawTimeInterval_.ix[:,0]/100000.0
    point = point.append(pd.Series([int(rawTimeInterval_.ix[rawTimeInterval_.shape[0] - 1,1])/100000.0]))
    proportion = np.cumsum(point/sum(point))
    phoneTimes = [round(x) for x in list(proportion * neuralSignalLen)]
    
    return phoneTimes


def NeuralSignalSegmentation(phoneTimes_, neuralSignal_, timeInterval_):
    phoneSegmentation_ = {} # dict{word: [{phone1: signalData}, {phone2: signalData},......]}
    for phoneIdx in range(timeInterval_.shape[0]):
        signalData = neuralSignal_[phoneTimes_[phoneIdx]:(phoneTimes_[phoneIdx + 1] - 1)]
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


def Generator(dataSeries, featureList):
    if 'mean' in featureList:
        if len(dataSeries) != 0:
            mean_ = float(unitFeature.Mean(dataSeries))
        else:
            mean_ = None
    else: 
        mean_ = None
    
    return [mean_]


def FeatureDF(featureList, sentences, ecogSlice, timeIntervals):
    # Create python data for timeIntervals
    #pickle.dump(timeIntervals, open('data/ALIGN.p',"wb"))
    #align = pickle.load(open("F:\BCM\SENTENCE\AUDIODATA\ALIGN.p","rb"))
    
    colName = ["phone name"]
    for i in featureList: colName.append(i) 
    
    totalFeatureDf = pd.DataFrame(columns = colName)
    for senIdx in range(0,len(sentences)):
        '''Create the feacture matrix for each sentence'''
        sentenceAdj = util.SentenceAdjustment(sentences[senIdx])
        # Ecog data
        ecogSeries = ecogSlice[str(senIdx)]
        # Ecog's phone split point
        timeInterval = timeIntervals[sentenceAdj] 
        phoneSplits = SplitPoint(timeInterval, len(ecogSeries))
        # Ecog's phone interval 
        phoneSegmentation = NeuralSignalSegmentation(phoneSplits, ecogSeries, timeInterval)
        # Feature matrix creation
        sentenceFeatureDf = pd.DataFrame(columns = colName)
        phoneIdx = 0
        for word in phoneSegmentation:
            wordSegmentation = phoneSegmentation[word]
            for phone in wordSegmentation:
                obsValue = Generator(phone["signalData"], featureList)
                obsValue.insert(0, phone["phone"])
                sentenceFeatureDf.loc[phoneIdx] = obsValue
                phoneIdx += 1
        totalFeatureDf = totalFeatureDf.append(sentenceFeatureDf)
        
    return totalFeatureDf

def FeatureNodeFreq(ecogTrain, sentenceTrain, nodeIdx, frequency, featureList, timeIntervals):
    '''Generate the features for the selected frequencies and nodes'''
    
    frequencyName = ['Delta', 'Theta', 'Alpha' ,'Beta' ,'Low Gamma', 'High Gamma']
    freqIdx = [frequencyName.index(freq) for freq in frequency]
    
    featureDF = pd.DataFrame()
    nameInd = 0
    for nodeIdx_ in nodeIdx:
        for freqIdx_ in freqIdx:
            ecogSlice = util.EcogSlicing(ecogTrain, nodeIdx_, freqIdx_)
            subFeatureDF = FeatureDF(featureList, sentenceTrain, ecogSlice, timeIntervals)
            if nameInd == 0:
                featureDF['phone name'] = subFeatureDF['phone name']
                
            colname = [str(nodeIdx_) + " " + frequency[freqIdx_] + " " + feat for feat in featureList]
            featureDF[colname] = subFeatureDF.drop(['phone name'], axis = 1)
            
    return featureDF

    