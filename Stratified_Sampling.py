import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#seaborn
color = sns.color_palette()
sns.set_style('darkgrid')

#sklearn
housing = pd.read_csv('D:/practical-data-science-master/VKHCG/01-Vermeulen/01-Retrieve/01-EDS/02-Python/housing_dataset.csv')
print(housing.head())
print(housing.info())

#Creating a heatmap of the attributes in the dataset

correlation_matrix = housing.corr(numeric_only=True)
plt.subplots(figsize=(8,6))
sns.heatmap(correlation_matrix,center=0,annot=True,linewidths=.3)

corr = housing.corr(numeric_only=True)
print(corr['median_house_value'].sort_values(ascending=False))

sns.displot(housing.median_income)
plt.show()
