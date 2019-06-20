#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:08:17 2019

@author: xiangshuyang
"""


import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import KNN 

def data(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index=0 
    fr = open(filename)
    for line in fr:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

def persondata(rangedata,datamin):
    video = float(input(\
                 "percentage of time playing video games?"))
    fly = float(input("frequent flier miles per year?"))
    icecream = float(input("liters of ice cream per year?"))
    inpP=np.array([5000,0.1,0.1])
    inpP= (inpP-datamin)/rangedata
    return inpP

def tellperson(filename):
    dataset,label= data(filename)
    normaldata,datamin,datarange = KNN.normalize(dataset)
    personal=persondata(datarange,datamin)
    classresult= KNN.classify(personal,label,dataset,3)
    return classresult
    

classresult=tellperson('datingdata.txt')
print(classresult)
    



        