import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/studentscores.csv')
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/4, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

# ---------------------- 그래프 1 (훈련 데이터) ----------------------
plt.figure(figsize=(10, 4))  # 첫 번째 그래프
plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째 그래프
plt.scatter(X_train, Y_train, color='red', label="Training Data")  # 훈련 데이터
plt.plot(X_train, regressor.predict(X_train), color='blue', label="Regression Line")  # 회귀선
plt.xlabel("Hours Studied")
plt.ylabel("Scores Obtained")
plt.title("Training Data")
plt.legend()

# ---------------------- 그래프 2 (테스트 데이터) ----------------------
plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째 그래프
plt.scatter(X_test, Y_test, color='red', label="Test Data")  # 테스트 데이터
plt.plot(X_test, regressor.predict(X_test), color='blue', label="Regression Line")  # 회귀선
plt.xlabel("Hours Studied")
plt.ylabel("Scores Obtained")
plt.title("Test Data")
plt.legend()

# 그래프 출력
plt.tight_layout()  # 자동으로 레이아웃 조정
plt.show()