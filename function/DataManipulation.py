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
    for sil in phoneSegmentation['sil']:
        if sil['signalData'].index[0] == 0 :
            silence = silence.append(sil['signalData'])
    silenceMean = float(np.mean(silence))
    ecogSeriesScaled = (ecogSeries - silenceMean)/silenceMean
    
    return ecogSeriesScaled
