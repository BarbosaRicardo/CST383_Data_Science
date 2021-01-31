# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:26:41 2021

@author: Dan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:23:18 2021

@author: Dan
"""

import numpy as np
# Const Vars
BOY = 1
GIRL = 2
ONE_GIRL = 3
TWO_GIRLS = 4
SAMPLE_SIZE = 10_000

#%%
## Lab3 #6

child1 = np.random.choice(2, SAMPLE_SIZE)+1

#%%
## Lab3 #7
child2 = np.random.choice(2, SAMPLE_SIZE)+1

#%%
## Lab3 #8

one_girl = np.sum((child1 + child2) >= (ONE_GIRL))  # faster
#og = np.sum((child1 == 2) | (child2 == 2)) # more readable

#%%
## Lab3 #9

both_girls = np.sum((child1 + child2) == (TWO_GIRLS)) # faster
#bg = np.sum((child1 == GIRL) & (child2 == GIRL))  # more readable

#%%
## Lab3 #10
# print((both_girls/SAMPLE_SIZE) / (one_girl/SAMPLE_SIZE))
print("P(two girls | one girl) =", both_girls / one_girl)

#%%
## Lab3 #11
elder_girl = np.sum(child1 == GIRL)
print("P(two girls | there is an elder girl) =", both_girls / elder_girl)

#%%
## Lab3 #12
