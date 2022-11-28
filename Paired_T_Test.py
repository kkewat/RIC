# Paired t test

from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
Input = 'F:/MSC/blood_pressure.csv'
df = pd.read_csv(Input)

print(df[['bp_before','bp_after']].describe())
#Checking any Significiant outliers

df[['bp_before','bp_after']].plot(kind='box')
#this save the file as png file
plt.savefig('F:/MSC/boxplot_outliers.png')

#Make a histogram to differences between the two scores.
df['bp_difference']=df['bp_before']-df['bp_after']
df['bp_difference'].plot(kind='hist',title='Blood Pressure Difference Histogram')
plt.savefig('F:/MSC/blood Pressure difference histogram.png')

stats.probplot(df['bp_difference'],plot=plt)
plt.title('Blood Pressure Difference Q-Q plot')
plt.savefig('F:/MSC/B-P-D_Q_Q_Plot.png')

stats.shapiro(df['bp_difference'])
stats.ttest_rel(df['bp_before'],df['bp_after'])
