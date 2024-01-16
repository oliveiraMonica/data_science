import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
## Assignment 1: EDA

1. Read in `income.csv`
2. Convert the target, `SalStat` into a binary numeric variable called `target`, and build a bar chart that plots the frequency of each value.
3. Explore the numeric features using histograms or boxplots.
4. Explore the categorical features using bar charts.
5. Consider writing functions for steps 3 and 4.
'''

# 1 - Read file
income = pd.read_csv("Data/income.csv")
income.head()

# 2 - Convert the target and build a bar chart
income["target"] = np.where(income["SalStat"] == ' less than or equal to 50,000', 0, 1)

income["target"].value_counts(normalize=True).plot.bar()
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Proportion')
plt.show()


# 3 - Explore the numeric features using histograms or boxplots.
def num_box_plotter(data):
    for column in data.select_dtypes("number"):
        sns.boxplot(data[column]).set(ylabel=column)
        plt.show()

#num_box_plotter(income)

# 4. Explore the categorical features using bar charts.
def cat_bar_plotter(data, normalize=False):

    for column in data.select_dtypes("object"):
        data[column].value_counts(normalize=normalize).plot.bar()
        plt.show()

cat_bar_plotter(income, normalize=True)