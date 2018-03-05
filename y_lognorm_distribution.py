#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:47:27 2018

@author: duxuewei
"""

from matplotlib import pyplot as plt
from scipy.stats import lognorm

plt.hist(train_y.values, bins=22, color = "#5cb29a")
scatter,loc,mean = lognorm.fit(train_y.values,
                               scale=train_y.mean(),
                               loc=0)
pdf_fitted = lognorm.pdf(np.arange(0,60,.5),scatter,loc,mean)
plt.plot(np.arange(0,60,.5),7500*pdf_fitted,'r', color = "#161491") 