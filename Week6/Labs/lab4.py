# %%
# 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# %%
# 2
df = pd.read_csv(
    "https://raw.githubusercontent.com/grbruns/cst383/master/College.csv", index_col=0)
# %%
# 3
df.info()
# %%
# 4.
# Build arrays to be used in training
x = df[['Outstate', 'F.Undergrad']].values
y = (df['Private'] == 'Yes').values.astype(int)  # Private will be set to 1?
# Does the training?
'''
** Research again ***
- Look up test_size
- Look up random_state
'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
# Wait, fit meant train, right? ** Research again **
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# %%
# 5. Print the first 10 rows of X_train to make sure the data is scaled.
x_train[:10, :]

# %%
# 6. Build a KNN classifier and train it.  Use the default value of k (which is parameter n_neighbors KNeighborsClassifier).  See lecture slides.
knn = KNeighborsClassifier()
# Oh, here is where we train!
knn.fit(x_train, y_train)
# %%
# 7. Make predictions using the training set, and save the predictions as variable 'predictions'.
predictions = knn.predict(x_test)
# %%
# 8. Compare the first ten predictions with the first ten correct (test set) values.
predictions[:10] == y_test[:10]
# %%
# 9. Which two variables do you need to compute the accuracy of your classifier on the test set?

# Predictions and y_test

# %%
# 10. Compute and print the test set accuracy of your classifier.
accuracy = (predictions == y_test).mean()
print(f'{accuracy}')
