# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:23:18 2021

@author: Dan
"""

import numpy as np
from fractions import Fraction
#%%
## Lab2 #8
rolls = np.array(np.random.choice(6,10000)+1)
fives = rolls[rolls == 5]
#ratio = fives.size/rolls.size
print(Fraction(fives.size,rolls.size))

#%%
## Lab2 # 9
die1 = np.random.choice(6, 10000) + 1
die2 = np.random.choice(6, 10000) + 1
sum_of_three =  np.sum(die1 + die2 == 3)
#ratio = sum_of_three/len(die1)
print(Fraction(sum_of_three, len(die1)))
