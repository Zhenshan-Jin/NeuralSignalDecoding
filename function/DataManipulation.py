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

def ScalingVisualCheck(ecogTrain, rawIntervals, expSentence, colIdx, plot):
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
        
        plt.axhline(y = scalingSil, c="yellow",linewidth=1,zorder=0, label = 'Scaling Silence')
        
        plt.xlabel('Phone')
        plt.ylabel('Mean')
        plt.title('Mean change after scaling')
        plt.xticks(index + bar_width, label)
        plt.legend()
         
        plt.tight_layout()
        plt.show()
    
    # Value check
    silZero = (expScale[label.index('sil')] < 1e-03)
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