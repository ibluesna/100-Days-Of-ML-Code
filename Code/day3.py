import pandas as pd
import numpy as np

dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values # R&D Spend, Administration, Marketing Spend, State
Y = dataset.iloc[:, 4].values # Profit

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder = 'passthrough'
)

X = np.array(column_transformer.fit_transform(X))
# print(X)
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

# 성능 평가 (R² Score)
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred)
print(f"R² Score: {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(Y_test, Y_pred, color='blue', label="Actual vs Predicted")
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label="Ideal Line")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit (Linear Regression)")
plt.legend()
plt.grid()
plt.show()