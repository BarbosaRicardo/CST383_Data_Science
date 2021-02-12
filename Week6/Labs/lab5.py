# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
# %%
df = pd.read_csv(
    'https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)

# %%
# 3. We will try to predict whether the tuition of a college based on the number of students from the top 10 percent of their high school class and the number of undergraduates..  Create a 2D NumPy array X from the 'Top10perc' and 'F.Undergrad' columns of df.

# Predictors
x = df[['Top10perc', 'F.Undergrad']].values
# %%
# 4. Create a 1D NumPy array y from the 'Outstate' column of df.

# Target
y = df['Outstate'].values

# %%
#  5. Split the data into training and test sets, with 30% of the data in the test set.  Use names X_train, y_train, X_test, y_test.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=23)
# %%

# 6. Now let's scale the values of the data.  It's important that we split that data before scaling.  We want to scale the training data and the test data based on the values in the training data.  By default, the StandardScaler class uses z-score normalization

scaler = StandardScaler()
# scaling x_train
x_train = scaler.fit_transform(x_train)
# scaling x_test
x_test = scaler.transform(x_test)

# %%
# 7. Build a KNN regressor and train it.  Use the default value of k (which is parameter n_neighbors KNeighborsRegressor).

knn = KNeighborsRegressor()
knn.fit(x_train, y_train)

# %%
# 8. Make predictions using the training set, and save the predictions as variable 'predictions'.

predictions = knn.predict(x_test)
# %%
# 9. Compare the first ten predictions with the first ten correct (test set) values.
print(predictions[:10])
print(y_test[:10])
predictions[:10] == y_test[:10]
# %%
# 10. Which two variables do you need to compute the mean squared error of your classifier on the test set?

'''
predictions & y_test
'''

# %%
# 11. Compute and print the mean squared error of your regressor.

MSE = ((predictions - y_test)**2).mean()
MSE
# %%
blind_mse = ((y_train.mean() - y_test)**2).mean()
blind_mse
# %%
