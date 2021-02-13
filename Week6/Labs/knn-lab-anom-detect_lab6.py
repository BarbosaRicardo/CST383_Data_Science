# -*- coding: utf-8 -*-
"""

kNN anomaly detection with application to college data

"""
# %%

import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)
# %%

# switch to seaborn default stylistic parameters
# see the very useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
sns.set_context('talk')

# change default plot size
rcParams['figure.figsize'] = 10, 8

#
# kNN anomaly detector
#

# compute distance between two n-dimensional points


def edist(x, y):
    return np.sqrt(np.sum((x-y)**2))

# return a distance matrix based on columns of float matrix x


def dist(x):
    m = x.shape[0]
    dm = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            dm[i, j] = edist(x[i, :], x[j, :])
            dm[j, i] = dm[i, j]
    return dm
# %%

# return indexes of anomalous elements of data frame df
# df - a data frame in which all columns are numeric
# k  - k parameter of KNN
# threshold - number of standard deviations past mean distance
#      to be considered an anomaly


def knn_anomalies(df, k=3, threshold=3):
    # scale the data and compute distance matrix
    x = df.apply(zscore).values
    dm = dist(x)

    # Compute and return an array containing the
    # row numbers of data frame df for which the distance to
    # the kth nearest neighbor is more than 'threshold' number
    # of standard deviations above the mean of the distances.
    # Pseudocode:
    # 1. for each row, compute distance to kth neareset neighbor
    # 2. use zscale-normalization on the distances
    # 3. get the indexes of the scaled distances that are larger than threshold

    #
    # YOUR CODE HERE
    #
    sorted_distances = np.apply_along_axis(np.sort, axis=1, arr=dm)
    dist_to_k_nearest = sorted_distances[:, k]
    indices = np.argsort(-dist_to_k_nearest)
    threshold = np.std(dist_to_k_nearest) * threshold
    num_anomalies = np.sum(dist_to_k_nearest > threshold)
    return indices[:num_anomalies]


#
# test the anomaly detector
#

# read the college data

#
df = pd.read_csv(
    'https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)
#


# compute the anomalies
features = ['Outstate', 'F.Undergrad']
anoms = knn_anomalies(df[features], k=7, threshold=6)

# add new 'Anomaly' column
x = np.full(len(df), False)
x[anoms] = True
df['Anomaly'] = x

# plot the colleges, showing anomalies
sns.scatterplot('Outstate', 'F.Undergrad', data=df,
                hue='Anomaly', style='Private')
# %%
# 5. Read through and run the code in the file to show anomalous colleges.  The code considers only features ‘Outstate’ and ‘F.Undergrad’.


def get_anomalies(df, k, threshold):
    features = ['Outstate', 'F.Undergrad']
    anoms = knn_anomalies(df[features], k, threshold)

    # add new 'Anomaly' column
    x = np.full(len(df), False)
    x[anoms] = True
    df['Anomaly'] = x

    return df['Anomaly']

# %%


get_anomalies(df, k=2, threshold=4)
sns.scatterplot('Outstate', 'F.Undergrad', data=df,
                hue='Anomaly', style='Private')

# %%

get_anomalies(df, k=10, threshold=3)
sns.scatterplot('Outstate', 'F.Undergrad', data=df,
                hue='Anomaly', style='Private')

# %%

df['Anamoly'] = get_anomalies(df, k=7, threshold=9)
sns.scatterplot('Outstate', 'F.Undergrad', data=df,
                hue='Anomaly', style='Private')

# %%

# 6. Examine one of the more anomalous colleges you discover.  Google it and see if you can find anything strange about it.

print(df.loc[df['Anamoly'] == True, 'Grad.Rate'])

'''
I set k and threshold to 10, and began to tapper it back until I had at least one anomaly show up.
Brigham Young University at Provo showed up. I researched this university and found that they boast to be the number 1 University in student engagement. However, when looking at their graduation avarage they have a 33 while the nation average is around 60. They have many full time graduates, and yet a very low graduation rate.
'''
# %%
