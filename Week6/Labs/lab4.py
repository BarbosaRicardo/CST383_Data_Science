# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# %%
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/College.csv", index_col=0)
# %%
