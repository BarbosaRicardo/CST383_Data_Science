import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% 2
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv')

# %% 
# 3. Visually explore the data frame using the Variable explorer tab in Spyder, which can be found in the upper right pane.

# explored

# %% 
# 4. Which Pandas commands can you use to get a quick overview of the data?

print(df.info())
print(df.describe())

# %% 
# 5. Remove the 'usid' and 'fnlwgt' columns from the data frame.

df.drop(columns=['usid','fnlwgt'], inplace=True)
df.info()


# %% 
# 6. Use a Pandas command to look at the first rows of the data frame.

df.head()

# %% 
# 7. The ‘education_num’ column records the number of years of education.  Use ‘describe’ to find min, max, median values for education_num.  Plot education_num using a histogram.  Label the x axis with 'years of education'.

print(df['education_num'].describe())

plt.hist(df['education_num'],bins=np.arange(0,18,2))
plt.xlabel('Years of Education')
plt.show()
# %%
# 8. Does it make sense to use education_num with a histogram?  Try it, and compare with a plot using a bar plot of the count of the rows by education_num (as shown in lecture).
data = df['education_num'].value_counts()
data.plot.bar()
plt.title('Counts of years of education')
plt.xlabel('Years of Education')
plt.ylabel('Counts')

'''
I think the historgram makes more senes. We can visually see the average, and below/above average. The counts just shows us the counts without substance. 
'''
# %%
# 9. Plot capital_gain with a density plot.  Did you find anything interesting?  Save your plot to a png file.
data = df['capital_gain']
sns.kdeplot(data)
plt.xlabel('Capital Gain')
plt.savefig('capita_gain_density_plot.png')

# %%
# 10. Investigate attribute ‘workclass’.  Plot it in an appropriate way.

data = df['workclass'].value_counts()
data.plot.bar()
plt.title('Counts by profession')
plt.xlabel('Profession')
plt.ylabel("Couns")
# %%
# 11. Use a bar plot to show the distribution of attribute ‘sex’.  Label the 'Male' and 'Female' bars with the fraction of rows associated with each sex (not a count).  Comment on what you find.  Why do you think the distribution is like this?

data = df['sex'].value_counts() / df.index.size
data.plot.bar()
plt.title("Distribution of female and male")
plt.
# %%
# 12. Use a horizontal bar plot to visualize attribute marital_status.
data = df['marital_status'].value_counts()
data.plot.barh()
plt.title("Marital status")
plt.xlabel("Counts")
plt.ylabel("Status")
# %%
# 13. If you have time, visualize all the attributes you haven’t explored yet.  Be sure to include 'native_country'.

# %%
# 14. If you still have time, do single-variable visualization for the attributes in the contribution campaign data.
