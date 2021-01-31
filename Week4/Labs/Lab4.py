# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:47:41 2021

@author: Dan
"""

import numpy as np
import scipy.stats as sts
import seaborn as sns
import matplotlib.pyplot as plt

#%% #6
def norm_plot(mean, sd):
    x = np.linspace(mean-3*sd, mean+3*sd)
    probs = sts.norm.pdf(x, loc=mean, scale=sd)
    plt.plot(x, probs, color="darkorange")
    
norm_plot(5,0.5)

#%% #7
# loc = mean & scale = std // makes sense...
samples = np.random.normal(loc=5,scale=0.5,size=1000)
sns.histplot(samples)

#%% #8
x = np.linspace(-5,5)
p = sts.norm.pdf(space_between)
# x = np.random.normal(10,0.2, size=1000)
# p = np.mean(np.absolute(x-10)<0.4)

#%% #9
plt.plot(x, p, color="blue")

#%% #10
std_val = np.std(space_between)
print("Witin 1 std:", -std_val, "to", std_val )

#%% #11

