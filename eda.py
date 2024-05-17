import pandas as pd
from lib import encoding as en, outliers as out, summary as sum, graphic as gra
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)
df = pd.read_csv('dataset/bank_dataset.csv')

sum.check_df(df)

result = out.grab_col_names(df)
cat_cols, num_cols, cat_but_car = result[0], result[1], result[2]

sum.cat_summary(df, cat_cols)

for col in cat_cols:
    sum.target_summary_with_cat(df, 'churn', col)

for col in num_cols:
    sum.target_summary_with_num(df, 'churn', col)

sum.correlation_matrix(df, num_cols)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
sns.boxplot(data=df['age'], ax=ax[0][0])
sns.boxplot(data=df['tenure'], ax=ax[0][1])
sns.boxplot(data=df['estimated_salary'], ax=ax[0][2])
sns.boxplot(data=df['balance'], ax=ax[1][0])
sns.boxplot(data=df['credit_score'], ax=ax[1][1])

out.for_check(df, df.columns)

sum.rare_analyser(df, 'churn', cat_cols=cat_cols)

gra.plot_numerical_col(df, num_cols)
gra.plot_categoric_col(df, cat_cols=cat_cols)

sizes = [df.churn[df['churn'] == 1].count(), df.churn[df['churn'] == 0].count()]
labels = ['Churned', 'Not Churned']
colors = ['red', 'orange']

plt.pie(sizes, labels=labels, autopct='%.2f%%', colors=colors)
plt.legend(loc='upper left')
plt.title("Churned VS Not Churned", size=10)
plt.show()
