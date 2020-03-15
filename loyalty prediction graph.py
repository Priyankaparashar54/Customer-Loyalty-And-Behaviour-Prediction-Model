import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
submission =pd.read_csv(r'C:\Users\Priyanka parashar\Desktop\Code\Code\data\submission.csv')
submission.shape
submission.info()
submission.head()
plt.subplot(1, 1, 1)
sns.distplot(submission.target,fit =None)
plt.xlabel('Customer Loyality')
