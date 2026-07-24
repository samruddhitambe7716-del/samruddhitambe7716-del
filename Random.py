import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
print(url)
df=pd.read_csv(url)
print(df)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
