import os

PATH = r'C:\Users\styli\Desktop\calcLTV'

os.chdir(PATH)

#%%

from datatable import dt, fread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta

from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter

from hyperopt import hp, fmin, tpe, rand, SparkTrials, STATUS_OK, space_eval, Trials


#%%

df = pd.read_excel("data/OnlineRetail.xlsx")
df.head()


'''
[예제 데이터 열 설명]

InvoiceNo: 송장 번호
StockCode: 제품 번호
Description: 제품명
Quantity: 주문 수량
InvoiceDate: 주문 일자 및 시각 (datetime 형태)
UnitPrice: 단가 (화폐 단위: 파운드 £)
CustomerID: 고객 번호
Country: 고객 거주 국가
'''

#%%
# RFM + T 연산 / 최근 구매 일(Recency), 몇 번이나 (Frequency), 누적 구매 금액(Mometary), 시간 (Time)
# BG/NBD : 고객이 앞으로 얼마나 구매할 것인가 (R / F / T)
# Gamma-Gammma : 고객이 앞으로 얼마를 구매할 것인가. (R / F / M)
# LTV : 고객이 구매할 횟수(BG/NBD) x 고객의 구매 금액(gamma-gamma) 로 계산


# InvoiceDate (주문 일자): Datetime -> date형
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date

# CustomerID: NULL인 것 제외
df = df[pd.notnull(df['CustomerID'])] 

# Quantity (주문 수량): 1 이상인 것
df = df[(df['Quantity'] > 0)] 

# Sales (구매 금액) 변수 생성
df['Sales'] = df['Quantity'] * df['UnitPrice']

# 고객 번호, 주문 일자, 구매 금액만 남기고 지우기
cols_of_interest = ['CustomerID', 'InvoiceDate', 'Sales']
df = df[cols_of_interest]

print('정제되어야 할 데이터의 형태')
print(df.head())
print('='*20)

#%%

HOLDOUT_DAYS = 90 # 마지막 90일은 Test 기간으로 정의
CURRENT_DATE = df['InvoiceDate'].max()
CALIBRATION_END_DATE = current_date - timedelta(days = HOLDOUT_DAYS)


PREDICT_MONTH = 12                # 앞으로 예측할 기간 (개월)
PREDICT_DATE = 30 * PREDICT_MONTH # 앞으로 예측할 기간 (일)
DISCOUNT_RATE = 0.01 # 예측할 기간동안, 앞으로 구매 금액에 할인율 적용 = 예측 금액 * (할인율)^(경과기간) + ....

#%%

# 전체 데이터 기준 RFMT 데이터
metrics_df = summary_data_from_transaction_data(df
                                          , customer_id_col = 'CustomerID'
                                          , datetime_col = 'InvoiceDate'
                                          , monetary_value_col='Sales'
                                          , observation_period_end=CURRENT_DATE)

# 전체 데이터를, Train / Valid 분할한 데이터
metrics_cal_df = calibration_and_holdout_data(df
                                          ,customer_id_col = 'CustomerID'
                                          ,datetime_col = 'InvoiceDate'
                                          ,calibration_period_end=CALIBRATION_END_DATE # train 데이터 기간
                                          ,observation_period_end=CURRENT_DATE         # 끝 기간
                                          ,monetary_value_col='Sales')

# BG/NBD 모델의 규칙을 위반한, 반복구매가 없는 데이터 제외
whole_filtered_df = metrics_df[metrics_df.frequency > 0]
filtered_df       = metrics_cal_df[metrics_cal_df.frequency_cal > 0]







'''
# Train / Valid 분할을 하지 않을 경우, 다음과 같이 데이터 프레임 생성.
summary_data_from_transaction_data(df,
                                   customer_id_col = 'CustomerID',
                                   datetime_col = 'InvoiceDate',
                                   monetary_value_col='Sales',
                                   observation_period_end=current_date)
'''


# from lifetimes.utils import * 에서 추가되었음.
print('lifetimes.utils 아래 함수로 정제된 결과')
print(metrics_cal_df.head())
print('='*20)

# F : 구매횟수
# R : 마지막 구매일 - 첫 구매일
# T : 집계일 - 첫 구매일
# M : 일 평균 구매 액수

#%%

# 평가 지표: default는 MSE
def score_model(actuals, predicted, metric='mse'):

    metric = metric.lower()

    # MSE / RMSE
    if metric=='mse' or metric=='rmse':
        val = np.sum(np.square(actuals-predicted))/actuals.shape[0]
    elif metric=='rmse':
        val = np.sqrt(val)
    # MAE
    elif metric=='mae':
        val = np.sum(np.abs(actuals-predicted))/actuals.shape[0]
    else:
        val = None

    return val

# BG/NBD 모형 평가
def evaluate_bgnbd_model(param):

    data   = inputs
    l2_reg = param

    # 모형 적합
    model = BetaGeoFitter(penalizer_coef=l2_reg)
    model.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])

    # 모형 평가
    frequency_actual = data['frequency_holdout']
    frequency_predicted = model.predict(data['duration_holdout']
                                        , data['frequency_cal']
                                        , data['recency_cal']
                                        , data['T_cal']
                                       )
    mse = score_model(frequency_actual, frequency_predicted, 'mse')

    return {'loss': mse, 'status': STATUS_OK}

# Gamma/Gamma 모델 평가
def evaluate_gg_model(param):

    data   = inputs
    l2_reg = param

    # GammaGamma 모형 적합
    model = GammaGammaFitter(penalizer_coef=l2_reg)
    model.fit(data['frequency_cal'], data['monetary_value_cal'])

    # 모형 평가
    monetary_actual = data['monetary_value_holdout']
    monetary_predicted = model.conditional_expected_average_profit(data['frequency_holdout'], data['monetary_value_holdout'])
    mse = score_model(monetary_actual, monetary_predicted, 'mse')

    # return score and status
    return {'loss': mse, 'status': STATUS_OK}


#%%
# fmin 모델을 활용한, 최소값 찾기
# BG/NBD모델의 l2 페널티 최소값
# l2 정규화 항은 0.001 ~ 0.1 사이가 적절함 (경험적으로 확인됨) - 단, 확인은 전 구간에 대해 실시

search_space = hp.uniform('l2', 0.0, 1.0) # l2 값의 최소-최대는 0~1
algo = tpe.suggest
trials = Trials()
inputs = filtered_df

argmin = fmin(
  fn = evaluate_bgnbd_model, # 목적함수
  space = search_space,      # 파라미터 공간
  algo = algo,               # 최적화 알고리즘: Tree of Parzen Estimators (TPE)
  max_evals=100,             # 반복수
  trials=trials            
  )

l2_bgnbd = space_eval(search_space,argmin)
print(f'BG/NBD Model l2 penalty : {l2_bgnbd}')

#%%
# Gamma-Gamma 모델의 l2 페널티 최소값

trials = Trials()

# GammaGamma
argmin = fmin(
  fn = evaluate_gg_model,
  space = search_space,
  algo = algo,
  max_evals=100,
  trials=trials
  )

l2_gg = space_eval(search_space,argmin)
print(f'Gamma-Gamma Model l2 penalty : {l2_gg}')



#%%
# 얻은 파라미터 값으로 재 훈련 실시.
# BG/NBD 모델

lifetimes_model = BetaGeoFitter(penalizer_coef=l2_bgnbd) #l2_bgnbd = hyperopt로 나온 결과
# calibration 데이터의 R,F,T로 모형 적합
lifetimes_model.fit(filtered_df['frequency_cal'], filtered_df['recency_cal'], filtered_df['T_cal']) 

# holdout 데이터로 모델 평가: F의 실제값과 예측값의 MSE
frequency_actual = filtered_df['frequency_holdout']
frequency_predicted = lifetimes_model.predict(filtered_df['duration_holdout']
                                    ,filtered_df['frequency_cal']
                                    , filtered_df['recency_cal']
                                    , filtered_df['T_cal'])
mse = score_model(frequency_actual, frequency_predicted, 'mse')

print('구매 일수에 대한 평균 제곱 오차 (MSE)')
print(f'MSE: {mse}')
print('='*20)

print('모델의 모수 설명')
print(lifetimes_model.summary)


# BG/NBD 모델의 정의
# https://playinpap.github.io/ltv-practice/
# 일정한 단위 시간 (T) 동안의 구매 횟수는 Pois(lambda * T)
# 여기서 lambda는 lambda ~ Gamma(r, alpha) 분포를 가지고, 이는 단위시간당 구매 횟수를 의미

# 이탈율은 (p)로 정의하고, 이탈할때까지 구매 횟수는 Geo(p) 분포
# 여기서 p는 p~Beta(alpha, beta)분포를 따른다

# lambda와 p는 서로 독립이다.


# 고객별 lambda (구매율) 의 분포
from scipy.stats import gamma, beta
import matplotlib.pyplot as plt

coefs = lifetimes_model.summary['coef']
x = np.linspace (0, 2, 100) 
y = gamma.pdf(x, a=coefs['r'], scale=1/coefs['alpha']) # BG/NBD에서의 모수 alpha는 scale 모수가 아닌 rate 모수이므로 역수!

plt.plot(x, y)
plt.rc('font', family='Malgun Gothic')
plt.title('단위시간(T) 당 유저의 구매 횟수 (포아송 분포의 Lambda, lambda ~ Gamma(r, alpha))')
plt.show()


coefs = lifetimes_model.summary['coef']
x = np.linspace(0, 1, 100) 
y = beta.pdf(x, a=coefs['a'], b=coefs['b']) # BG/NBD에서의 모수 alpha는 scale 모수가 아닌 rate 모수이므로 역수!

plt.plot(x, y)
plt.rc('font', family='Malgun Gothic')
plt.title('고객의 이탈 확률 (기하 분포의 p, p ~ Beta(a, b))')
plt.show()


#%%
# 얻은 파라미터 값으로 재 훈련 실시.
# Gamma-Gamma 모델
spend_model = GammaGammaFitter(penalizer_coef=l2_gg)
spend_model.fit(filtered_df['frequency_cal'], filtered_df['monetary_value_cal'])

# conditional_expected_average_profit: 고객별 평균 구매 금액 예측
monetary_actual = filtered_df['monetary_value_holdout']
monetary_predicted = spend_model.conditional_expected_average_profit(filtered_df['frequency_holdout']
                                                                    ,filtered_df['monetary_value_holdout'])

mse = score_model(monetary_actual, monetary_predicted, 'mse')

print('구매 금액에 대한 평균 제곱 오차 (MSE)')
print(f'MSE: {mse}')

bins = 100
plt.figure(figsize=(15, 5))

plt.hist(monetary_actual, bins, label='actual', histtype='bar', color='STEELBLUE', rwidth=0.99)
plt.hist(monetary_predicted, bins, label='predict', histtype='step', color='ORANGE',  rwidth=0.99)
plt.legend(loc='upper right')
plt.rc('font', family='Malgun Gothic')
plt.title('고객의 구매 금액 (실제 / 예측)')
plt.show()




#%%
# LTV를 연산

# 이용하는 모델 : Train / Valid 분할 후 측정된 모델
# 예측하는 데이터 : 전체 데이터 기반 예측

final_df = whole_filtered_df.copy()

# LTV 연산
final_df['ltv'] = spend_model.customer_lifetime_value(lifetimes_model,
                                                     final_df['frequency'],
                                                     final_df['recency'],
                                                     final_df['T'],
                                                     final_df['monetary_value'],
                                                     time = PREDICT_MONTH,
                                                     discount_rate = DISCOUNT_RATE # monthly discount rate ~12.7% 연간
                                                     )

# 예상 구매 횟수
final_df['predicted_purchases'] = lifetimes_model.conditional_expected_number_of_purchases_up_to_time(PREDICT_DATE
                                                                                      , final_df['frequency']
                                                                                      , final_df['recency']
                                                                                      , final_df['T'])
# 예상 구매 금액
final_df['predicted_monetary_value'] = spend_model.conditional_expected_average_profit(final_df['frequency']
                                                                    ,final_df['monetary_value'])



#%%

print('LTV 예측 결과')
print(final_df)


print('LTV 최상위 5명 - 매출은 설정한 예측기간 참조')
final_df.sort_values(by="ltv").tail(5)
