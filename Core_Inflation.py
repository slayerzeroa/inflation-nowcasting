import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

core_cpi = [0.02, 0.04, 0.03, 0.02, 0.06, 0.03, 0.02, 0.05, 0.04, 0.04, 0.03, 0.02]
core_pce = [0.01, 0.02, 0.01, 0.02, 0.02, 0.01, 0.01, 0.03, 0.02, 0.02, 0.01, 0.01]

X = np.array(core_cpi).reshape(-1, 1)
y = np.array(core_pce).reshape(-1, 1)

lr = LinearRegression(fit_intercept=True) ## 절편항 있는 회귀 모형, False 절편항 제외
lr.fit(X, y) ## 모형 적합

print('절편 :', lr.intercept_)
print('회귀 계수 :', lr.coef_)

X_test = np.array([[0.01]])
print(lr.predict(X_test)) # 예측

print(lr.score(X, y))


def repeat_data(data, repeat):
    result = []
    for i in data:
        for j in range(repeat):
            result.append(i)
    return result

core_cpi = repeat_data(core_cpi, 30)
print(core_cpi)