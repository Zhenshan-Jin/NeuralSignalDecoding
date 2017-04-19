#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:11:09 2017
@author: zhenshan
"""

import pandas as pd
import numpy as np
import pickle
# parallel moduel
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
# user-defined
import SentenceSegmentation
import FeatureGeneration
# visualization
import matplotlib.pyplot as plt




def ScalingBySilence(ecogTrain, rawIntervals):
    '''Scaling data frame by calculating the percentage change compared to sil mean'''
    try:
        ecogTrainScaled = pickle.load( open( "data/ecogTrainScaled.p", "rb" ) )
    except IOError:
        timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)# create split point data frame
        ecogTrainScaled = {}
        for sent in ecogTrain:
            timeInterval = timeIntervals[sent] 
            phoneSplits = SentenceSegmentation.SplitPoint(timeInterval, ecogTrain[sent].shape[0])
            # Parallel Computing 
            ScalingHelperPartial = partial(ScalingHelper, phoneSplits_ = phoneSplits, timeInterval_ = timeInterval)
            partResult = Parallel(n_jobs=mp.cpu_count())(delayed(ScalingHelperPartial)(col) for colName, col in ecogTrain[sent].iteritems())
            ecogTrainScaled[sent] = pd.concat(partResult, axis = 1)
            
        pickle.dump(ecogTrainScaled, open( "data/ecogTrainScaled.p", "wb" ))
        
    return ecogTrainScaled

def ScalingBySilenceNew(ecogNew, ecogSilence, rawIntervals):
    '''Scaling data frame by calculating the percentage change compared to sil mean'''
    try:
        ecogNewScaled = pickle.load( open( "data/NewData/ecogNewScaled.p", "rb" ) )
    except IOError:
        timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)# create split point data frame
        ecogNewScaled = {}
        for sent in ecogNew:
            try:
                ecogNewSpeech = ecogNew[sent].ix[150:,:]
                timeInterval = timeIntervals[sent] 
                ecogSilenceSent = ecogSilence[sent]
                phoneSplits = SentenceSegmentation.SplitPoint(timeInterval, ecogNewSpeech.shape[0])
                # Parallel Computing 
                ScalingHelperPartial = partial(ScalingHelperNew, ecogSilenceSent_ = ecogSilenceSent, phoneSplits_ = phoneSplits, timeInterval_ = timeInterval)
                partResult = Parallel(n_jobs=mp.cpu_count())(delayed(ScalingHelperPartial)(colName, col) for colName, col in ecogNew[sent].iteritems())
                ecogNewScaled[sent] = pd.concat(partResult, axis = 1)
            except:
                pass
        pickle.dump(ecogNewScaled, open( "data/NewData/ecogNewScaled.p", "wb" ))
        
    return ecogNewScaled


def ScalingHelper(ecogSeries, phoneSplits_, timeInterval_):
    '''Getting scaled data for each of 420 frequencies'''
    phoneSegmentation = SentenceSegmentation.NeuralSignalSegmentation(phoneSplits_, ecogSeries, timeInterval_)
    silence = pd.Series(name = ecogSeries.name)
#   Scaling by silence at the beginning
    silence = silence.append(phoneSegmentation['sil'][0]['signalData'])
#   Scaling by all the silence in the sentence
#    for sil in phoneSegmentation['sil']:
#        silence = silence.append(sil['signalData'])
    silenceMean = float(np.mean(silence))
    ecogSeriesScaled = 100 * (ecogSeries - silenceMean)/silenceMean
    
    return ecogSeriesScaled


def ScalingHelperNew(colNames, ecogSeries, ecogSilenceSent_, phoneSplits_, timeInterval_):
    '''Getting scaled data for each of 70 frequencies'''
#    phoneSegmentation = SentenceSegmentation.NeuralSignalSegmentation(phoneSplits_, ecogSeries, timeInterval_)
    silence = ecogSilenceSent_.ix[:,colNames]
#   Scaling by silence at the beginning
#    silence = silence.append(phoneSegmentation['sil'][0]['signalData'])
#   Scaling by all the silence in the sentence
#    for sil in phoneSegmentation['sil']:
#        silence = silence.append(sil['signalData'])
    silenceMean = float(np.mean(silence))
    ecogSeriesScaled = 100 * (ecogSeries - silenceMean)/silenceMean
    
    return ecogSeriesScaled


def PhoneLabeling(ecogTrainScaled, rawIntervals):
    timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)
    ecogTrainScaledLabeled = {}
    for sent, df in ecogTrainScaled.items():
        timeInterval = timeIntervals[sent] 
        phoneSplits = SentenceSegmentation.SplitPoint(timeInterval, df.shape[0])
        # Ecog's phone interval 
        tempSeries = df.ix[:,0]
        phoneSegmentation = SentenceSegmentation.NeuralSignalSegmentation(phoneSplits, tempSeries, timeInterval)
        
        # Feature matrix creation
        colNames = ["phoneIndex", "phoneName", "timeIndex"]
        sentenceFeatureDf = pd.DataFrame(columns = colNames)
        phoneIdx = 0
        base = 0
        for word in phoneSegmentation:
            wordSegmentation = phoneSegmentation[word]
            subPhoneIdx = 0
            for phone in wordSegmentation:
                sentenceFeaturedfTemp = pd.DataFrame()
                sentenceFeaturedfTemp["phoneIndex"] = [phoneIdx] * phone["signalData"].size
                phoneName = ''.join([i for i in phone["phone"] if not i.isdigit()])
                sentenceFeaturedfTemp["phoneName"] = [phoneName] * phone["signalData"].size
                sentenceFeaturedfTemp["timeIndex"] = phone["signalData"].index.values
                if subPhoneIdx == 0 and phone == 'sil':
                    base = phone["signalData"].index.values.tolist()[0]
                sentenceFeatureDf = sentenceFeatureDf.append(sentenceFeaturedfTemp)
                subPhoneIdx += 1
                phoneIdx += 1
        sentenceFeatureDf["timeIndex"] = sentenceFeatureDf["timeIndex"] - base
        sentenceFeatureDf = sentenceFeatureDf.reset_index()
        df = df.reset_index()
        del sentenceFeatureDf["index"]
        del df["index"]

        sentenceFeatureDf = pd.concat([sentenceFeatureDf,df], axis = 1)
        ecogTrainScaledLabeled[sent] = sentenceFeatureDf        
        
    return ecogTrainScaledLabeled


def PhoneLabelingNew(ecogNewScaled, rawIntervals):
    timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)
    ecogTrainScaledLabeled = {}
    for sent, df in ecogNewScaled.items():
        dfSpeech = df.ix[150:, :]
        timeInterval = timeIntervals[sent] 
        phoneSplits = SentenceSegmentation.SplitPoint(timeInterval, dfSpeech.shape[0])
        # Ecog's phone interval 
        tempSeries = dfSpeech.ix[:,0]
        phoneSegmentation = SentenceSegmentation.NeuralSignalSegmentation(phoneSplits, tempSeries, timeInterval)
        
        # Feature matrix creation
        colNames = ["phoneIndex", "phoneName", "timeIndex"]
        sentenceFeatureDf = pd.DataFrame(columns = colNames)
        phoneIdx = 0
        base = 0
        for word in phoneSegmentation:
            wordSegmentation = phoneSegmentation[word]
            subPhoneIdx = 0
            for phone in wordSegmentation:
                sentenceFeaturedfTemp = pd.DataFrame()
                sentenceFeaturedfTemp["phoneIndex"] = [phoneIdx] * phone["signalData"].size
                phoneName = ''.join([i for i in phone["phone"] if not i.isdigit()])
                sentenceFeaturedfTemp["phoneName"] = [phoneName] * phone["signalData"].size
                sentenceFeaturedfTemp["timeIndex"] = phone["signalData"].index.values
                if subPhoneIdx == 0 and phone == 'sil':
                    base = phone["signalData"].index.values.tolist()[0]
                sentenceFeatureDf = sentenceFeatureDf.append(sentenceFeaturedfTemp)
                subPhoneIdx += 1
                phoneIdx += 1
        sentenceFeatureDf["timeIndex"] = sentenceFeatureDf["timeIndex"] - base
        sentenceFeatureDFNonSpeech = pd.DataFrame(dict(zip(colNames,[[-1 for i in range(150)],['silNoneSpeech' for i in range(150)],[i for i in range(150)]])))
        sentenceFeatureDf = pd.concat([sentenceFeatureDFNonSpeech, sentenceFeatureDf], axis = 0)
        sentenceFeatureDf = sentenceFeatureDf.reset_index()
        dfSpeech = dfSpeech.reset_index()
        del sentenceFeatureDf["index"]
        del dfSpeech["index"]
        
        # Append non speech sil to the data frame
        sentenceFeatureDf = sentenceFeatureDf.sort_values(by = 'timeIndex').reset_index()
        del sentenceFeatureDf["index"]
        # Create right phone order
        preIndex = None
        index = -2
        newIndex = []
        for i in sentenceFeatureDf['phoneIndex']:
            if i != preIndex:
                index += 1
                preIndex = i
            newIndex.append(index)
            
        sentenceFeatureDf['phoneIndex'] = newIndex
        
        sentenceFeatureDf = pd.concat([sentenceFeatureDf,df], axis = 1)
        ecogTrainScaledLabeled[sent] = sentenceFeatureDf        
        
    return ecogTrainScaledLabeled


def ExportScalingData(ecogTrainScaledLabled):
    '''export data to default data folder ./data/'''
    frames = list()
    sentences = pd.DataFrame(columns = ['sentence'])
    sentIndex = 0
    for sent, df in ecogTrainScaledLabled.items():
        df["sentenceIndex"] = np.array([sentIndex] * df.shape[0])
        frames.append(df)
        sentences = sentences.append({'sentence':sent}, ignore_index=True)
        sentIndex += 1
    
    result = pd.concat(frames)
    cols = result.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    
    result = result[cols]
    dataFilePath = "data/ecogTrainScaledLabeled.csv"
    result.to_csv(dataFilePath,  index=False)
    
    sentences.index.name  = 'index'
    sentenceFilePath = "data/sentence_N_IndexScaledLabled.txt"
    sentences.to_csv(sentenceFilePath)
    

def ExportScalingDataNew(ecogNewScaledLabled):
    '''export data to default data folder ./data/NewData'''
    frames = list()
    sentences = pd.DataFrame(columns = ['sentence'])
    sentIndex = 0
    for sent, df in ecogNewScaledLabled.items():
        df["sentenceIndex"] = np.array([sentIndex] * df.shape[0])
        frames.append(df)
        sentences = sentences.append({'sentence':sent}, ignore_index=True)
        sentIndex += 1
    
    result = pd.concat(frames)
    cols = result.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    
    result = result[cols]
    dataFilePath = "data/NewData/ecogNewScaledLabeled.csv"
    result.to_csv(dataFilePath,  index=False)
    
    sentences.index.name  = 'index'
    sentenceFilePath = "data/NewData/sentence_N_IndexScaledLabled.txt"
    sentences.to_csv(sentenceFilePath, index = False, header = False)



