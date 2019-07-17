#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:39:25 2019

@author: xiangshuyang
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:00:51 2019
@author: xiangshuyang
"""


import numpy as np 
import operator

def classify(inpX,label,data,k):
    m,n=np.shape(data)
    xmat=np.tile(inpX,(m,1)) #every colum of xmat is equal to inpX
    diff= np.sqrt(np.sum((xmat-data)**2,axis=1))
    sortindex=diff.argsort()
    classcount={} 
    for i in range(k):
        votelabel = label[sortindex[i]] # vote for the label 
        classcount[votelabel] = classcount.get(votelabel,0) + 1    
    classcountsort=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return classcountsort[0][0]


def normalize(data):
    m,n=np.shape(data)
    datamin=data.min(0)
    datamax = data.max(0)
    datarange= datamax-datamin
    normaldata=(data-np.tile(datamin,(m,1)))/np.tile(datarange,(m,1))
    return normaldata,datamin,datarange