
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')
# sns.set_palette('pastel')
# %%
df = pd.read_csv(
    "https://raw.githubusercontent.com/grbruns/cst383/master/College.csv", index_col=0)

# %%
# 3. Derive a new column, ‘Size’, from the F.Undergrad variable.  The possible values of Size should be “small”, “medium”, or “large”.   The value “small” should be assigned to the colleges in the “bottom 3rd” of F.Undergrad values, “medium” should be assigned to the “middle 3rd”, and “large” to the “top 3rd”.  Use the Pandas ‘quantile’ function to find the corresponding F.Undergrad values.  (If you're not sure how to do this, see the hints right away).

breaks = df['F.Undergrad'].quantile([0, 0.33, 0.66, 1.0])
df['Size'] = pd.cut(df['F.Undergrad'], include_lowest=True,
                    bins=breaks, labels=['small', 'medium', 'large'])

# %%
#  4. Use the faceting (also known as 'conditioning') idea to create three scatter plots, one for each value of your new variable size.  The scatterplot should show PhD on the x axis and Outstate on the y axis.  Try to make your plot look approximately like this:

g = sns.FacetGrid(df, col="Size", height=4, aspect=0.8)
g.map(plt.scatter, 'PhD', 'Outstate', color='orange')

# Look at the plots for a while and think their significance.   Is the plot for large colleges different from the plot for small colleges?  What does this say about large and small colleges.  Also, do you see any interesting outliers?

'''
The plot for larger colleges, seems to be a little tighter. There also tends to be more faculty with PhD's than in the other colleges. 
'''
# %%
# 5. Repeat problem 5, but this time show a single scatterplot, with color used to distinguish small, medium, and large schools.  Your plot might look something like this:

sns.scatterplot(data=df, x='PhD', y='Outstate', hue='Size')

# Look at the plot you created for this problem compared to the last problem.  Which do you think is easier to interpret.   Spend some time on this -- it's important.

'''
I think this second plot is easier to interpret. Instead of looking at different graphs
and comparing each one, we can just look at one and easily see the changes. However, with this graph
it is harder to see the areas where the most dots are concentrated in an individual graph.
'''

# %%
# 6. Repeat the last plot, but this time use both shape and color to indicate college size.  Do you think your new plot is easier to interpret than your plot of the previous problem?

sns.scatterplot(data=df, x='PhD', y='Outstate',
                hue='Size', size='Size', style='Size')

'''
# It feels a lot messier, I don't know that I would say it's easier to interpret.
'''
# %%
# 7. Create three violin plots, showing the distribution of tuition at each of the three college size values.  My plot looks like this:

# sns.violinplot(data=df, x='Size', y='Outstate', color='#1f77b4')
sns.catplot(y='Outstate', col='Size', data=df, kind='violin', height=4, aspect=0.7)

# %%

# 8. Repeat the last plot, but now show the raw data on the plot.

# sns.violinplot(data=df, x='Size', y='Outstate', color='#1f77b4', inner='stick')
sns.catplot(y='Outstate', col='Size', data=df, kind='violin', inner='stick', height=4, aspect=0.7)

# %%

