#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:36:49 2017

@author: zhenshan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SentenceSegmentation
import DataManipulation
import FeatureGeneration


def ScaledDataTestNew(ecogNewScaled):
    '''test for the new scaled data'''
    nonSpeechMean = []
    for sent in ecogNewScaled:
        for colName, col in ecogNewScaled[sent].iteritems():
            nonSpeechMean.append(float(np.mean(col[0:149])))
    plt.hist(nonSpeechMean)
    plt.title('Mean diftribution of non-speech ecog signal for %s senetence' %len(ecogNewScaled))
    
    
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
    expScaledSeries = DataManipulation.ScalingHelper(expSeries, phoneSplits, timeInterval)
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