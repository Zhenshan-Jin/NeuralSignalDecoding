#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:40:46 2017

@author: zhenshan
"""

def SentenceAdjustment(sentence):
    words = sentence.split()
    j = 0
    sentenceAdj = ""
    while j < len(words):
        if j < len(words) - 1: 
            sentenceAdj = sentenceAdj + words[j] + "_"
        else:
            sentenceAdj = sentenceAdj + words[j] + "AV_RMS"
        j += 1
        
    return sentenceAdj

def EcogSlicing(ecogDic, freqIdx, nodeIdx):
    sliceIdx = freqIdx * 70 + nodeIdx
    idx = 0
    ecogSlice = {}
    for sent in ecogDic:
        ecogSlice[str(idx)] = ecogDic[sent].iloc[:,[sliceIdx]]
        idx += 1
    return ecogSlice
