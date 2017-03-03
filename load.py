#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:48:44 2017

@author: zhenshan
"""

import os
import pandas as pd
import util

def loadPhone():
    # Reading data from directory
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)
    try:
        rawData_ = open('AlignedTime.txt').read()
        os.chdir(owd)
    except IOError:
        os.chdir(owd)
    
    return rawData_

def loadOrderedSentence():
    # Reading the sentences which have the same order as nueral signal
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)
    try:
        sentence = open('TRAIN.txt',"r").readlines()
        os.chdir(owd)
    except IOError:
        os.chdir(owd)

    return sentence

def loadEcog():
    '''Create dictionary with key: sentence name; value: ecog data frame'''
    owd = os.getcwd()
    dataDir = owd + '/data'
    os.chdir(dataDir)
    try:
        breakPoint = [int(index) for index in open('train_breakpoint.txt').readlines()]
        breakPoint.insert(0,0)
        train_X_ecog = pd.read_csv('train_X_ecog.csv', header=None)
        os.chdir(owd)
        sentences = loadOrderedSentence()
    except IOError:
        os.chdir(owd)

    ecog = {}
    for idx in range(len(breakPoint) - 1):
        sentence = util.SentenceAdjustment(sentences[idx])
        ecog[sentence] = train_X_ecog[breakPoint[idx] : breakPoint[idx + 1]]
        
    return ecog