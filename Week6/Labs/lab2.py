# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# 2. load data
df = pd.read_csv(
    'https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)


# %%
'''
3. 
Create scatter plots to compare some of the variables.  Here are some questions to help you get started.  Follow your curiosity.
Do smaller colleges spend more?  (variables F.Undergrad and Expend)
Do smaller colleges charge more?  (variables F.Undergrad and Outstate)
Define two new variables: perc.accept and perc.enroll.  The first is the percentage of students who accepted out of those who applied (use variables Accept and Apps).  The second is the percentage of students who enrolled out of those where were accepted (use variables Enroll and Accept).
   Now use the new variables in scatter plots.  For example: are more selective colleges more expensive, generally?

   For each scatter plot that you create, write something about the form/shape, the direction, and the strength.  Think about the possible meaning of the scatterplot.

   Don't just stick to my suggestions, choose some of your own variables to explore.  Think of questions that you find interesting. 
'''
sns.pairplot(df, vars=['Expend', 'F.Undergrad', 'Outstate'])
# %%

df['perc.accept'] = df['Accept'] / df['Apps']
df['perc.enroll'] = df['Enroll'] / df['Accept']
sns.pairplot(df, vars=['perc.accept', 'perc.enroll', 'Books', 'Room.Board'])

# %% [markdown]
'''
- Seems like room.board tends to be lower based on at colleges with a lower percentage of enrollment
- Cost of books isn't affected by any of the variables above. 
- 
'''

# %%
# 4.4. Think of more questions and see if you can create plots to help understand them.  For example, are more selective colleges more expensive?  Just create more scatterplots and explore them.  There are lots of things to pursue in the college data.

# %%
# Do colleges with higher acceptance spend more on their students?
sns.scatterplot(data=df, x='perc.accept', y='Expend')
'''
This does not appear to be the case. There are a few outliers that doe spend more with a lower acceptance rate.
'''
mask = (df['perc.accept'] < 0.3) & (df['Expend'] > 30_000)
df[mask]
# As suspected, they are the ivy league schools.

# %%
# Do schools who spend more on their students have students that donate more?
sns.scatterplot(data=df, x='Expend', y='perc.alumni')
# This doesn't seem to be the case either
# %%
# Are students who spend more from out of state more or less likely to donate?
sns.scatterplot(data=df, x='Outstate', y='perc.alumni')
# It seems that the more they spend from in out-of-state tuition, the more likely they are to donate!
# There is a pretty strong positive correlation.
df[['Outstate', 'perc.alumni']].corr()
# %%
