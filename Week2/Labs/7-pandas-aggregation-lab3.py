# -*- coding: utf-8 -*-
"""
Pandas dataframes

@author: Glenn Bruns
"""
import numpy as np
import pandas as pd


# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# =============================================================================
# Read data
# =============================================================================

# read 1994 census summary data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv")
df.set_index('usid', inplace=True)
df.drop('fnlwgt', axis=1, inplace=True)

# =============================================================================
# Simple aggregation
# =============================================================================

# print the average age

# get the min, max, and avg value for each numeric column

# for a dataframe you get the aggregate for each column by default

# =============================================================================
# Aggregation with grouping
# =============================================================================

# how many people in each category of education?
# Try using pandas function value_counts().

# for each native country, what is the average education num?

# repeat the previous problem, sorting the output by the average
# education num value

# for each occupation, compute the median age

# repeat the previous problem, but sort the output

# find average hours_per_week for those with age <= 40, and those with age > 40
# (this uses something labeled 'advanced' in the lecture)

# do the same, but for age groups < 40, 40-60, and > 60

# get the rows of the data frame, but only for occupations
# with an average number of education_num > 10
# Hint: use filter

# =============================================================================
# Vectorized string operations
# =============================================================================

# create a Pandas series containing the values in the native_country column.
# Name this series 'country'.

# how many different values appear in the country series?

# create a Series containing the unique country names in the series.
# Name this new series 'country_names'.

# modify country_names so that underscore '_' is replaced
# by a hyphen '-' in every country name.  Use a vectorized operation.
# (See https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)

# modify country_names to replace 'Holand' with 'Holland'.

# modify country_names so that any characters after 5 are removed.
# Again, use a vectorized operation

# Suppose we were to use only the first two letters of each country.
# Write some code to determine if any two countries would then share
# a name.

# If you still have time, write code to determine which countries
# do not have a unique name when only the first two characters are
# used.  Hint: look into Pandas' Series.duplicated().

# =============================================================================
# Handling times and dates
# =============================================================================

# read gas prices data
gas = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/Gasoline_Retail_Prices_Weekly_Average_by_Region__Beginning_2007.csv")

# create a datetime series and make it the index of the dataset

# plot the gas prices for NY city

# plot gas prices for NY city and Albany on the same plot

# if you still have time, see if you can find and plot California
# gas prices



