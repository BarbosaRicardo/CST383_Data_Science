# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:10:24 2019

@author: Glenn
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# %%

# read the data
df = pd.read_csv(
    "https://raw.githubusercontent.com/grbruns/cst383/master/german-credit.csv")
bad_loan = df['good.loan'] - 1

# %%

# use only numeric data, and scale it
df = df[["duration.in.months", "amount", "percentage.of.disposable.income", "at.residence.since",
         "age.in.years", "num.credits.at.bank"]]
X = df.apply(zscore).values
y = bad_loan.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

# see how knn classifier works as training size changes

# %%

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
    X_train1 = X_train[:tr_size, :]
    y_train1 = y_train[:tr_size]

    # train model on a subset of the training data
    knn.fit(X_train1, y_train1)

    # error on subset of training data
    tr_predicted = knn.predict(X_train1)
    err = (tr_predicted != y_train1).mean()
    tr_errs.append(err)

    # error on all test data
    te_predicted = knn.predict(X_test)
    err = (te_predicted != y_test).mean()
    te_errs.append(err)

'''
 3. Add code to plot the learning curve.  On the x axis you will have the training set size; on the y axis you will have one line for classification error with the training set, and another line for classification with the test set.  Use different colors for the training and test error lines, and include a legend
'''
#
# plot the learning curve here
#

plt.plot(tr_sizes, te_errs)
plt.plot(tr_sizes, tr_errs)
plt.title(f'Learning curve')
plt.xlabel('Training sizes')
plt.ylabel('Error')
plt.legend(['Testing Error', 'Training Error'])
# %%

'''
4.	Extend the code by adding a loop so that you produce learning curves for k = 1, 3, 5, and 9.  Put all the plots on one page.  Put the value of k in the title of the learning curve plot.
'''

df = df[["duration.in.months", "amount", "percentage.of.disposable.income", "at.residence.since",
         "age.in.years", "num.credits.at.bank"]]
X = df.apply(zscore).values
y = bad_loan.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


ks = [1, 3, 5, 9]
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    te_errs = []
    tr_errs = []
    tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
    for tr_size in tr_sizes:
        X_train1 = X_train[:tr_size, :]
        y_train1 = y_train[:tr_size]

        # train model on a subset of the training data
        knn.fit(X_train1, y_train1)

        # error on subset of training data
        tr_predicted = knn.predict(X_train1)
        err = (tr_predicted != y_train1).mean()
        tr_errs.append(err)

        # error on all test data
        te_predicted = knn.predict(X_test)
        err = (te_predicted != y_test).mean()
        te_errs.append(err)

    plt.plot(tr_sizes, te_errs)
    plt.plot(tr_sizes, tr_errs)
    plt.title(f'Learning curve k={k}')
    plt.xlabel('Training sizes')
    plt.ylabel('Error')
    plt.legend(['Testing Error', 'Training Error'])
    plt.show()

print('done')

'''
5. Explain the curves you get.  For example, what’s with the low training error when k = 1?

The curves appear really jagged to me, but it seems that the learning curve for k=9 seems to have the least variance. 
The training error rate for k=1 appears to be flatlined at 0. I feel like that shouldn't happen.

6. What do the learning curves tell you?   Write some sentences to explain what the learning curves tell you about bias and variance.

There appears to be lots of variance. If this is the case, then I need more data.
'''
# %%
'''
7. Add some more features, and produce the 4 learning curve plots again.  Explain how and why the learning curves changed.
'''
# read the data - again
df = pd.read_csv(
    "https://raw.githubusercontent.com/grbruns/cst383/master/german-credit.csv")
bad_loan = df['good.loan'] - 1
print(df.info())

df = df[["duration.in.months", "amount", "percentage.of.disposable.income", "at.residence.since",
         "age.in.years", "num.credits.at.bank", "good.loan", "num.dependents"]]
X = df.apply(zscore).values
y = bad_loan.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

ks = [1, 3, 5, 9]

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    te_errs = []
    tr_errs = []
    tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
    for tr_size in tr_sizes:
        X_train1 = X_train[:tr_size, :]
        y_train1 = y_train[:tr_size]

        # train model on a subset of the training data
        knn.fit(X_train1, y_train1)

        # error on subset of training data
        tr_predicted = knn.predict(X_train1)
        err = (tr_predicted != y_train1).mean()
        tr_errs.append(err)

        # error on all test data
        te_predicted = knn.predict(X_test)
        err = (te_predicted != y_test).mean()
        te_errs.append(err)

    plt.plot(tr_sizes, te_errs)
    plt.plot(tr_sizes, tr_errs)
    plt.title(f'Learning curve k={k}')
    plt.xlabel('Training sizes')
    plt.ylabel('Error')
    plt.legend(['Testing Error', 'Training Error'])
    plt.show()

print('done')

'''
I added more predictors which helped reduce the variance, but now it seem like there is more bias and testing is now lower than the training error.
'''

# %%
