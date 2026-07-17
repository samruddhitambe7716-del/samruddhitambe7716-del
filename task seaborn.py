import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=sns.load_dataset('tips')
print(df)
print(df.head())
sns.scatterplot(x='total_bill',y='tip',data=df)
plt.show()
sns.lineplot(x='total_bill',y='tip',data=df)
plt.show()
sns.barplot(x='total_bill',y='tip',data=df)
plt.show()
sns.countplot(x='day',data=df)
plt.show()
sns.histplot(df['total_bill'],bins=10)
plt.show()
sns.kdeplot(df['total_bill'],fill=True)
plt.show()

