import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def repeat_data(data, repeat):
    result = []
    for i in data:
        for j in range(repeat):
            result.append(i)
    return result

def cal_MA(data, window):
    MA = np.mean(data[-window+1:])
    return MA

def cal_change(d1, d2):
    change = (d1 - d2) / d2
    return change



core_cpi = [0.04, 0.10, -0.03, 0.01, 0.06, 0.03, 0.02, 0.05, 0.04, 0.04, 0.03, 0.02]
core_pce = [0.01, 0.02, 0.01, 0.02, 0.02, 0.01, 0.01, 0.03, 0.02, 0.02, 0.01, 0.01]
cpi = [0.09, 0.12, 0, 0.02, 0.04, 0.05, 0.02, 0.01, 0.05, 0.04, 0.01, 0.04]
pce = [0.02, 0.03, 0.05, 0.04, 0.07, 0.03, 0.02, 0.04, 0.01, 0.02, 0.02, 0.02]

core_cpi, core_pce = repeat_data(core_cpi, 30), repeat_data(core_pce, 30)
cpi, pce = repeat_data(cpi, 30), repeat_data(pce, 30)

gasoline_timeline = [60 for i in range(14*30+1)] + np.cumsum(np.random.normal(0, 3, 14*30+1))
oil_timeline = [70 for i in range(14*30+1)] + np.cumsum(np.random.normal(0, 3, 14*30+1))
food_cpi = [30 for i in range(14*30+1)] + np.cumsum(np.random.normal(0, 3, 14*30+1))
food_off_pce = [20 for i in range(14*30+1)] + np.cumsum(np.random.normal(0, 3, 14*30+1))

chg_gasoline = []
chg_oil = []
chg_food_cpi = []
chg_food_off_pce = []

for i in range(len(gasoline_timeline)-1):
    chg_gasoline.append(cal_change(gasoline_timeline[i+1], gasoline_timeline[i]))
    chg_oil.append(cal_change(oil_timeline[i+1], oil_timeline[i]))
    chg_food_cpi.append(cal_change(food_cpi[i + 1], food_cpi[i]))
    chg_food_off_pce.append(cal_change(food_off_pce[i + 1], food_off_pce[i]))



# X = np.array(core_cpi[0:360]).reshape(-1, 1)
# y = np.array(core_pce[0:360]).reshape(-1, 1)
#
# lr = LinearRegression(fit_intercept=True)  ## 절편항 있는 회귀 모형, False 절편항 제외
# lr.fit(X, y)  ## 모형 적합
#
# predict_core_cpi = cal_MA(core_cpi[0:360], 12*30)
# core_cpi.append(predict_core_cpi)
#
# predict_core_pce = lr.predict(np.array([[predict_core_cpi]]))[0][0]
# core_pce.append(predict_core_pce)
#
#
# # CPI 회귀
# X = np.transpose(np.array([np.array(core_cpi[0:360]), np.array(gasoline_timeline[0:360]),np.array(food_cpi[0:360])]).reshape(3, -1))
# y = np.array(cpi[0:360]).reshape(-1, 1)
#
# lr = LinearRegression(fit_intercept=True)  ## 절편항 있는 회귀 모형, False 절편항 제외
# lr.fit(X, y)  ## 모형 적합
#
# predict_cpi = lr.predict(np.array([[predict_core_cpi, gasoline_timeline[360], food_cpi[360]]]))[0][0]
# cpi.append(predict_cpi)
#
# # PCE 회귀
# X = np.transpose(np.array([np.array(core_pce[0:360]), np.array(gasoline_timeline[0:360]),np.array(food_off_pce[0:360])]).reshape(3, -1))
# y = np.array(pce[0:360]).reshape(-1, 1)
#
# lr = LinearRegression(fit_intercept=True)  ## 절편항 있는 회귀 모형, False 절편항 제외
# lr.fit(X, y)  ## 모형 적합
#
# predict_pce = lr.predict(np.array([[predict_core_pce, gasoline_timeline[360], food_off_pce[360]]]))[0][0]
# pce.append(predict_pce)




for i in range(59):
    # core PCE 회귀
    X = np.array(core_cpi[i:360+i]).reshape(-1, 1)
    y = np.array(core_pce[i:360+i]).reshape(-1, 1)

    lr = LinearRegression(fit_intercept=True)  ## 절편항 있는 회귀 모형, False 절편항 제외
    lr.fit(X, y)  ## 모형 적합

    predict_core_cpi = cal_MA(core_cpi[i:360+i], 12*30)
    core_cpi.append(predict_core_cpi)

    predict_core_pce = lr.predict(np.array([[predict_core_cpi]]))[0][0]
    core_pce.append(predict_core_pce)

    # CPI 회귀
    X = np.array([np.array(core_cpi[i:360+i]), np.array(chg_gasoline[i:360+i]),np.array(chg_food_cpi[i:360+i])]).reshape(-1, 3)
    y = np.array(cpi[i:360+i]).reshape(-1, 1)

    lr = LinearRegression(fit_intercept=True)  ## 절편항 있는 회귀 모형, False 절편항 제외
    lr.fit(X, y)  ## 모형 적합

    predict_cpi = lr.predict(np.array([[predict_core_cpi, chg_gasoline[361+i], chg_food_cpi[361+i]]]))[0][0]
    cpi.append(predict_cpi)

    # PCE 회귀
    X = np.array([np.array(core_pce[i:360+i]), np.array(chg_gasoline[i:360+i]),np.array(chg_food_off_pce[i:360+i])]).reshape(-1, 3)
    y = np.array(pce[i:360+i]).reshape(-1, 1)


    lr = LinearRegression(fit_intercept=True)  ## 절편항 있는 회귀 모형, False 절편항 제외
    lr.fit(X, y)  ## 모형 적합

    predict_pce = lr.predict(np.array([[predict_core_pce, chg_gasoline[361+i], chg_food_off_pce[361+i]]]))[0][0]
    pce.append(predict_pce)


plt.figure(figsize=(20, 10))
plt.plot(cpi, label='CPI')
plt.plot(pce, label='PCE')
plt.plot(core_cpi, label='Core CPI')
plt.plot(core_pce, label='Core PCE')

plt.legend()
plt.show()