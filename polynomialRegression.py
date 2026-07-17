import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from matplotlib import style
style.use('ggplot')

warnings.filterwarnings('ignore')

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\Samruddhi\Desktop\ts\position.csv")

# Display first 5 rows
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Scatter Plot
# -----------------------------
plt.figure(figsize=(12,6), dpi=200)
sns.scatterplot(x='Level', y='Salary', data=df)
plt.title("Salary vs Level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# -----------------------------
# Independent & Dependent Variables
# -----------------------------
x = df[['Level']].values
y = df['Salary'].values

# -----------------------------
# Simple Linear Regression
# -----------------------------
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

print("\nLinear Regression Score:", lin_reg.score(x, y))

pred1 = lin_reg.predict(x)

plt.figure(figsize=(12,6), dpi=200)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, pred1, color='red', label='Linear Regression')
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# -----------------------------
# Polynomial Regression Degree 2
# -----------------------------
from sklearn.preprocessing import PolynomialFeatures

poly2 = PolynomialFeatures(degree=2)
x_poly2 = poly2.fit_transform(x)

poly_reg2 = LinearRegression()
poly_reg2.fit(x_poly2, y)

pred2 = poly_reg2.predict(x_poly2)

plt.figure(figsize=(12,6), dpi=200)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, pred2, color='green', label='Degree 2')
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# -----------------------------
# Polynomial Regression Degree 3
# -----------------------------
poly3 = PolynomialFeatures(degree=3)
x_poly3 = poly3.fit_transform(x)

poly_reg3 = LinearRegression()
poly_reg3.fit(x_poly3, y)

pred3 = poly_reg3.predict(x_poly3)

plt.figure(figsize=(12,6), dpi=200)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, pred3, color='orange', label='Degree 3')
plt.title("Polynomial Regression (Degree 3)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# -----------------------------
# Polynomial Regression Degree 6
# -----------------------------
poly6 = PolynomialFeatures(degree=6)
x_poly6 = poly6.fit_transform(x)

poly_reg6 = LinearRegression()
poly_reg6.fit(x_poly6, y)

pred6 = poly_reg6.predict(x_poly6)

plt.figure(figsize=(12,6), dpi=200)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, pred6, color='purple', label='Degree 6')
plt.title("Polynomial Regression (Degree 6)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# -----------------------------
# Compare All Models
# -----------------------------
plt.figure(figsize=(12,6), dpi=200)

plt.scatter(x, y, color='black', label='Actual Data')
plt.plot(x, pred1, color='red', label='Linear Regression')
plt.plot(x, pred2, color='green', label='Degree 2')
plt.plot(x, pred3, color='orange', label='Degree 3')
plt.plot(x, pred6, color='purple', label='Degree 6')

plt.title("Polynomial Regression Comparison")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# -----------------------------
# Predict Salary for Level 6.5
# -----------------------------
level = np.array([[6.5]])

linear_salary = lin_reg.predict(level)
poly_salary = poly_reg6.predict(poly6.transform(level))

print("\nPredicted Salary (Linear Regression):", linear_salary[0])
print("Predicted Salary (Polynomial Degree 6):", poly_salary[0])