import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

gasoline_timeline = [60 for i in range(12*30)] + np.cumsum(np.random.normal(0, 1.5, 12*30))
oil_timeline = [70 for i in range(12*30)] + np.cumsum(np.random.normal(0, 0.5, 12*30))

food_timeline = [40 for i in range(12*30)] + np.cumsum(np.random.normal(0, 1.5, 12*30))

X = np.array(gasoline_timeline).reshape(-1, 1)
y = np.array(oil_timeline).reshape(-1, 1)
lr = LinearRegression(fit_intercept=True) ## 절편항 있는 회귀 모형, False 절편항 제외
lr.fit(X, y) ## 모형 적합

print('절편 :', lr.intercept_)
print('회귀 계수 :', lr.coef_)

X_test = np.array([[70]])
print(lr.predict(X_test)) # 예측

print(lr.score(X, y))
