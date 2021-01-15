# -*- coding: utf-8 -*-
"""
Pandas dataframes lab

Fill in your code after each comment (unless code already provided).

@author: Glenn Bruns
"""

import numpy as np
import pandas as pd

# here are the names of seven CSUMB students
students = ['Sean', 'Laura', 'Angel', 'Mariana', 'Austin', 'Jose', 'Ana']

# here are the MPG values for the students' cars
student_mileage = [21.1, 26.6, 16.3, 33.7, 31.2, 52.0, 27.1]

# create a Pandas dataframe df containing the MPG values, using the
# names for the index.  Use 'mpg' for the mileage column.

# print the dataframe you just created

# using df, print the mileage for Ana's car using the .loc attribute

# using df, print the rows of the dataframe where mpg > 30 (use a boolean mask)

# using df, print the mileage values for Laura and Austin using .loc

# here are distances the students travel to get to CSUMB
student_dist = {'Sean':8.1, 'Laura':5.4, 'Angel':12.8, 'Austin':15.0, 'Jose':22.2, 'Ana':18.5}

# create another dataframe df2 containing the distance values, and
# again use the names for the index.  Use 'distance' for the
# distance column.

# print the dataframe you get by adding 'distance' as a new column.  
# All students should appear (Mariana will have NaN for distance).
# Use pandas.join

# update df using your last answer so that it includes the 'distance' column

# the distance value for Mariana should be 'NaN'.  Update the value
# to 3.5 miles using .loc

# using df, show mpg and distance for students with distance > 20
# Don't use .loc or .iloc

# repeat, but using .loc

# print the underlying 2D numpy array containing the dataframe data
# (Hint: you use .index on a data frame to get the row index; how do
# you get the data?)

# add a new column, 'gas', which shows the number of
# gallons of gas needed for a round-trip drive to CSUMB

# print the first two rows of the data frame using .iloc

# If you still have time, redo the problem above that asks you
# to use pandas.join, but this time use pandas.merge.  You
# will want to rerun your earlier code so that df has no
# 'distance' column

# Repeat the problem you just solved, but use pandas.concat

# If you still have time, play more with some Pandas dataframes
# to get comfortable with indexing.  Also, play with other
# dataframe features discussed in lecture.
