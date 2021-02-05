# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:38:44 2021

@author: Dan
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pd.set_option('display.max_rows', 55)

#%%

# 1. Create an Python file in Spyder, and enter this code to read the data:

# infile = "https://raw.githubusercontent.com/grbruns/cst383/master/campaign-ca-2016-sample.csv" (Links to an external site.)

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/campaign-ca-2016-sample.csv")

#%%
# 2. Look at some of the data using the variable explorer in the upper right pane of Spyder. (Select the 'Variable explorer' tab and double click on 'df'.)

# * Clicked *

#%%

# 3. Look at the type of each column in df.  (We'll sometimes refer to the columns as 'variables', 'attributes', or 'features'.)  How many columns are shown as numeric?  Do you think some of the columns should be numeric but aren't?

df.info()

# Only 2 columns are numeric
# zip could potentially become numeric, but that depends on how it will be used because it may be easier to parse it as an object than as a numeric.

#%%

# 4. Which of the columns contain NA values?  Use Python to figure out the total number of NA values in the data set. 
df.isna()

print(f"Cols with NA: \n{df.isna().sum()}")
print(f"Total NA in dataset: {df.isna().sum().sum()}")

#%%

# 5. Can you find values (besides 'nan') that indicate missing data?  You can try doing this with Python or by searching manually through df.  High-level hint: you might expect an NA value in a column to appear many times.

df['contbr_occupation'].value_counts().head(20)
# Not employeed, None, and Retired may not have data
df['contbr_city'].value_counts()
# seems we only have all the values here
#%%

# 6. Does missing data exist in attribute contbr_employer?  If so, how is it encoded?  Would it make sense to change it?

df['contbr_employer'].value_counts().head(55)
df.loc[df['contbr_employer'].isna(), 'contbr_employer']
df[df['contbr_occupation'] == "NONE"]['contbr_employer']

# so far i see nan, NONE, Mr., Ms.

#%%

# 7. Look more at contbr_employer.  Do you see any other data quality issues?
emplrs = np.sort(df['contbr_employer'].fillna("#").unique())
# Found #Name?, 'None', 'Self', -, -NUN-
mask = df['contbr_employer'].str.contains("Employed", case=False, na=False)
df.loc[mask, 'contbr_employer'].unique()
# Found 16 different types of employers assosciated with the word "employed"
#%%

# 8. How many different values are there in attribute ‘memo_cd’?  What are the values?  What fraction of the values are empty?
print(f"Different values {df['memo_cd'].value_counts().sum()}")
print(f"Ratio of missing values: {df['memo_cd'].isna().sum() / df.index.size}")

# We only have about 3% of those values filled out.

#%%

# 9. Attribute ‘contb_receipt_amt’ is the amount of the contribution.  Produce a histogram of the values.  Be sure your plot has a good title and good axis labels.

data = df['contb_receipt_amt']
plt.hist(data)
plt.title("Contributions by Donor")
plt.xlabel("Contribution Amounts")
plt.ylabel("Contribution Donor Index")

#%%

# 10. What is the range of ‘contb_receipt_amt’ values?  Do any of them look suspicious?  How should you deal with negative campaign contributions?  Do negative contributions tend to be paired with positive contributions?
data.describe()
df.loc[df['contb_receipt_amt'] < 0, 'contb_receipt_amt'].value_counts()
cand_ids = df[df['contb_receipt_amt'] == -2700]['cand_id'].unique()
df.where(df['cand_id'] == cand_ids)
# cand_id_mask = df.loc[df['contb_receipt_amt'] < 0, 'cand_id'].unique().reshape(1,-1)
# cand_id_match = df.loc[df['cand_id'] == cand_id_mask, 'contb_receipt_amt']

#%%

# 11. Attribute contbr_zip has the zip code of a contributor.  Are all zip codes in the same format?  If not, do you think it would be appropriate to process the zip code data?
df["contbr_zip"].str.len().value_counts()
# They are not the same length
# we would need to strip the last 4 digits of the longer zipcodes

#%%

# 12. Create a histogram of the lengths of contbr_employer values (i.e., the length of the values as strings).  Is the distribution unusual?  Give an explanation, based on working with the data, of why some employer length values seem to be very popular.
x = df['contbr_employer'].str.len()
plt.hist(x)
plt.title("String Length of contbr_employer")
plt.ylabel("Frequency")
plt.xlabel("String Lengths")

#%%

# 13. If we scale a vector of numeric values using 0-1 scaling, then the smallest value in the vector will become 0, the largest will become 1, and the others will be scaled linearly between 0 and 1.  Create a new attribute, s_amt1, from ‘contb_receipt_amt’ by using 0-1 scaling. 

x = df['contb_receipt_amt'] 
df['s_amt1'] = (x - x.min()) / (x.max() - x.min())

#%%

# 14. What do memo_cd values mean?  How do they relate to the values in ‘memo_text’?  Note: an "earmarked" contribution is one that's not given directly to a candidate but marked to indicate the candidate to which the contribution will be given.
df.loc[df['memo_cd'] == 'X', 'memo_text'].value_counts()

# seems like the x in memo_cd means there was a change in the way the funds were used