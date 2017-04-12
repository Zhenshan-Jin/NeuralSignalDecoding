#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:12:17 2017

@author: zhenshan
"""

import pandas as pd
import numpy as np

def ReformatRawData():
    '''Reformat the data from 70 seperate nodes file into one .csv file'''
    try:
        sentDataFinal = pd.read_csv("data/NewData/X_ecog.csv", header = None)
        scalingSilenceFinal = pd.read_csv("data/NewData/X_ecog_sil.csv", header = None)
        breakPoint = pd.read_csv("data/NewData/breakpoint.txt", header = None)
    except IOError:
        breakPoint = []
        scalingSilence = []
        sentData = []
        
        for nodeIdx in range(70):
            dataPath = 'data/Sentences_Nodes/node_%s.csv'%(nodeIdx + 1)
            nodeData = pd.read_csv(dataPath, header = None)
            silence = nodeData.ix[:,0:149]
            scalingSilence.append(pd.DataFrame(silence.values.reshape(-1,1)))
            nonSilence = nodeData.ix[:,150:601].transpose()
            
            subNoneSilence = []
            for sentIdx in range(210):
                subSentData = nonSilence.ix[:,sentIdx].dropna()
                if nodeIdx == 0:
                    breakPoint.append(subSentData.size)
                subNoneSilence.append(subSentData.rename(nodeIdx))
            sentData.append(pd.concat(subNoneSilence, axis = 0))
        
        sentDataFinal = pd.concat(sentData, axis = 1)
        scalingSilenceFinal = pd.concat(scalingSilence, axis = 1)  
        breakPoint = np.array(breakPoint).cumsum()
        
        sentDataFinal.to_csv("data/NewData/X_ecog.csv", header = False, index = False)
        scalingSilenceFinal.to_csv("data/NewData/X_ecog_sil.csv", header = False, index = False)
        pd.Series(breakPoint).to_csv("data/NewData/breakpoint.txt", index = False)