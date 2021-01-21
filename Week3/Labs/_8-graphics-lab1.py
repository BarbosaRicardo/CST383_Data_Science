# -*- coding: utf-8 -*-
"""
Python graphics lab

@author: Glenn Bruns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# change default plot size
rcParams['figure.figsize'] = 6,4
# change plotting style
plt.style.use('seaborn')

# =============================================================================
# Read the 'mtcars' (Motor Trend cars) dataset
# =============================================================================
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/mtcars.csv")

# =============================================================================
# Plotting
# =============================================================================

# Write Python code to answer the following questions.  Try hard to answer
# the questions from memory before using the lecture slides or other resources.


# It takes practice to create good-looking plots.  It's a good idea to start
# with a simple plot and then to improve it if the it appears useful.

# First, look at the mtcars data in Spyder, and using df.info(), to get an
# idea of what is in the data set.  Some information about the data can be
# found here:
# https://www.rdocumentation.org/packages/datasets/versions/3.6.0/topics/mtcars

# (YOUR CODE HERE -- and similarly for further questions)

df.info()
#%%#
# Create a basic scatterplot showing each car’s mpg as a function of engine 
# displacement.  (in other words, mpg goes on the y axis and displacement on 
# the x axis.)  Use the matplotlib scatter() function.
plt.scatter(df['disp'], df['mpg'])

#%%
# Improve your plot by adding 'displacement (cu. inches)' as the x axis
# label, and 'miles per gallon' as the y axis label.
plt.scatter(df['disp'], df['mpg'])
plt.xlabel('displacement (cu. inches)')
plt.ylabel('miles per gallon')

#%%
# Make the dots dark red
plt.scatter(df['disp'], df['mpg'], color='red')
plt.xlabel('displacement (cu. inches)')
plt.ylabel('miles per gallon')

#%%
# Use upward triangles instead of circles
# (hint: use the 'marker' option in plt.scatter)
plt.scatter(df['disp'], df['mpg'], color='red', marker='^')
plt.xlabel('displacement (cu. inches)')
plt.ylabel('miles per gallon')

#%%
# Add a title
# Hint: in scatter plats, we often say 'A by B' when A is the
# thing on the y axis and B is the thing on the x axis.  For
# example: 'MPG by engine displacement'.
plt.scatter(df['disp'], df['mpg'], color='red', marker='^')
plt.xlabel('displacement (cu. inches)')
plt.ylabel('miles per gallon')
plt.title('MPG by engine displacement')

#%%
# Compute the average mpg, and add a black dotted horizontal
# line to show this value
plt.scatter(df['disp'], df['mpg'], color='red', marker='^')
plt.xlabel('displacement (cu. inches)')
plt.ylabel('miles per gallon')
plt.title('MPG by engine displacement')
plt.axhline(y=df['mpg'].mean(), linestyle='-')

#%%
# Color the points in the plot according to whether the transmission
# is manual or automatic.  Use blue if automatic (am = 0) and orange if manual (am = 1).
# Note: blue/orange are easy to distinguish for many people, even most
# people with some form of colorblindness.)
#colors = [ 'blue' if x == 0 else 'orange' for x in df['am'] ]
#plt.scatter(df['disp'], df['mpg'], marker='^', c=colors)
plt.scatter(df.loc[df['am']==0, 'disp'], df.loc[df['am']==0, 'mpg'], marker='^', color='darkorange')
plt.scatter(df.loc[df['am']==1, 'disp'], df.loc[df['am']==1, 'mpg'], marker='^', color='royalblue')
plt.xlabel('displacement (cu. inches)')
plt.ylabel('miles per gallon')
plt.title('MPG by engine displacement')
plt.axhline(y=df['mpg'].mean(), linestyle='-', color='black')

#%%
# Add a legend
# You may want to do this by splitting the data up, and then
# calling plt.scatter() twice, once for am=0 and once for am=1.
# Hint: use the 'label' parameter in plt.scatter(), and then plt.legend().
plt.scatter(df.loc[df['am']==0, 'disp'], df.loc[df['am']==0, 'mpg'], marker='^', color='darkorange', label='Automatic')
plt.scatter(df.loc[df['am']==1, 'disp'], df.loc[df['am']==1, 'mpg'], marker='^', color='royalblue', label='Manual')
plt.xlabel('displacement (cu. inches)')
plt.ylabel('miles per gallon')
plt.title('MPG by engine displacement')
plt.axhline(y=df['mpg'].mean(), linestyle='-', color='black')
plt.legend()

#%%
# Compute the number of cars for each possible value of number of cylinders, 
# and assign the result to variable 'cyl_counts'.
# Hint: you can use this code:  cyl_counts = df['cyl'].value_counts()
cyl_counts = df['cyl'].value_counts()

#%%
# Using matplot lib, make a bar plot showing the number
# of cars for each number of cylinders


#%%
# improve your plot by adding title, x and y axis labels, and the color
# of your choice

#%%
# Repeat the process, but this time use Seaborn to create your
# barplot
# Note: you can use plt.title() to give your plot a title

#%%
# Create a histogram of mpg values, using matplotlib, then
# seaborn, then pandas

#%%
# If you still have time, experiment with seaborn and pandas plots.
# For example, in pandas, what happens if you try to plot two columns
# of the mtcars data?


