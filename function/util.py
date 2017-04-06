#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:40:46 2017

@author: zhenshan
"""

def SentenceAdjustment(sentence):
    '''Modify the sentence format as in raw Ecog sentence data'''
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


def EcogSlicing(ecogDic, nodeIdx, freqIdx):
    '''Select the frequency&node col from total 420 cols for each sentence'''
    sliceIdx = freqIdx * 70 + nodeIdx
    ecogSlice = {}
    for sent in ecogDic:
        ecogSlice[sent] = ecogDic[sent].iloc[:,[sliceIdx]]
    return ecogSlice

