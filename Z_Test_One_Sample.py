from statsmodels.stats import weightstats as stests
import pandas as pd
from scipy import stats

df=pd.read_csv('F:/MSC/blood_pressure.csv')
df[['bp_before','bp_after']].describe()
print(df)
ztest,pval=stests.ztest(df['bp_before'],x2=None,value=156)
print(float(pval))
if pval<0.05:
    print('Reject Null Hypothesis')
else:
    print('Accept Null hypothesis')
