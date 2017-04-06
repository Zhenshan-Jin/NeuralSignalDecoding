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
    ecogSeriesScaled = (ecogSeries - silenceMean)/silenceMean
    
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


def ScalingVisualCheck(expSentence, ecogTrain, rawIntervals, isTotal, colIdx = None, plot = False):
    '''Check the accuracy for each sentence
       plot is True, then colIdx is necessary'''
    if isTotal:
        scalingTest = list()
        for colIdx in range(420):
            result = ScalingVisualCheckPerCol(ecogTrain, rawIntervals, expSentence, colIdx, plot)
            scalingTest.append(result)
        if scalingTest.count(True) == len(scalingTest):
            print('The scaling is correct')
        else:
            print('Error: The scaling is wrong')
            
        return scalingTest
    
        if plot:
            ScalingVisualCheckPerCol(ecogTrain, rawIntervals, expSentence, colIdx, plot)
            
    else:
        if plot:
            ScalingVisualCheckPerCol(ecogTrain, rawIntervals, expSentence, colIdx, plot)
        
        
        
def ScalingVisualCheckPerCol(ecogTrain, rawIntervals, expSentence, colIdx, plot):
    '''Cheking the scaling accuracy: if the original phone mean > first scaling sil mean: scaling phone mean > 0
                                     else: scaling phone mean < 0
       return: true/false, whether scaling correctly'''
    timeIntervals = SentenceSegmentation.AlignmentPoint(rawIntervals)# create split point data frame
    timeInterval = timeIntervals[expSentence] 
    phoneSplits = SentenceSegmentation.SplitPoint(timeInterval, ecogTrain[expSentence].shape[0])
    expSeries = ecogTrain[expSentence].ix[:,colIdx]
    expScaledSeries = ScalingHelper(expSeries, phoneSplits, timeInterval)
    phoneSegmentationOrig = SentenceSegmentation.NeuralSignalSegmentation(phoneSplits, expSeries, timeInterval)
    phoneSegmentationScaled = SentenceSegmentation.NeuralSignalSegmentation(phoneSplits, expScaledSeries, timeInterval)
    
    featureDfOrig = ScalingVisualCheck_Helper(phoneSegmentationOrig)
    featureDfScaled = ScalingVisualCheck_Helper(phoneSegmentationScaled)
    
    label = list(featureDfScaled.ix[:,0])
    
    expOrig = list(featureDfOrig.ix[:,1])
    expScale = list(featureDfScaled.ix[:,1])
    scalingSil = expOrig[label.index('sil')]

    if plot:
        # Graph check
        fig, ax = plt.subplots()
        index = np.arange(len(label))
        bar_width = 0.35
        opacity = 0.8
        
        scaleColor = ['red' if val > scalingSil else 'green' for val in expOrig]
        
        plt.bar(index, expOrig, bar_width,
                         alpha=opacity,
                         color= 'b',
                         label='Original')
        
        plt.bar(index + bar_width, expScale, bar_width,
                         alpha=opacity,
                         color=scaleColor,
                         label='Scaled')
        
        plt.axhline(y = scalingSil, c="black",linewidth=1,zorder=0, label = 'Scaling Silence')
        
        plt.xlabel('Phone')
        plt.ylabel('Mean')
        plt.title('Mean change after scaling')
        plt.xticks(index + bar_width, label)
        plt.legend()
         
        plt.tight_layout()
        plt.show()
    
    # Value check
    silZero = (expScale[label.index('sil')] < 1e-02)
    featurePN = list()
    for idx, val in enumerate(expScale):
        if(expOrig[idx] - scalingSil) * val >= 0:
            featurePN.append(True)
        else:
            featurePN.append(False)
    
    if featurePN.count(True) == len(featurePN) and silZero:
        return True # Not scaling correctly
    else:
        return False


def ScalingVisualCheck_Helper(phoneSegmentation_):
    colName = ["phone name"]
    featureList = ['mean'] 
    for i in featureList: colName.append(i) 

    sentenceFeatureDf = pd.DataFrame(columns = colName)
    phoneIdx = 0
    for word in phoneSegmentation_:
        wordSegmentation = phoneSegmentation_[word]
        for phone in wordSegmentation:
            obsValue = FeatureGeneration.FeatureGenerator(phone["signalData"], featureList)
            obsValue.insert(0, phone["phone"])
            sentenceFeatureDf.loc[phoneIdx] = obsValue
            phoneIdx += 1
            
    return sentenceFeatureDf.dropna()