import numpy as np
import pandas as pd

dataset = pd.read_csv('../datasets/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = float("nan"), strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# categorical_features → 제거됨.
# ColumnTransformer를 사용하여 특정 열을 OneHotEncoder로 변환.
# remainder="passthrough"를 설정하여 나머지 열은 그대로 유지.
from sklearn.compose import ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],  # 범주형 열을 OneHotEncoder로 변환
    remainder="passthrough"  # 나머지 열은 그대로 유지
)
X = column_transformer.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)
X_test = sc_X.transform(X_test)

# Print Data
# 최종 데이터 출력
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("Y_train:\n", Y_train)
print("Y_test:\n", Y_test)