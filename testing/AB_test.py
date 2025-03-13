import os

PATH = r'C:\Users\styli\Desktop\testing'
DATA_PATH_LOGIT = './data/logistic_dataset.csv'
DATA_PATH_NORM = './data/normal_dataset.csv'
DATA_PATH_GAMMA = './data/gamma_dataset.csv'

os.chdir(PATH)

#%%
import numpy as np
import pandas as pd

# Testing
import scipy.stats as stats
from scipy.stats import chi2_contingency
from statsmodels.stats import proportion

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns

# Logic
from logic.LogisticTesting import OnesideLogisticTesting, TwosideLogisticTesting
from logic.Testing import OnesideTesting, TwosideTesting

# Make Dataset
from data.BuildDataset import make_data_frame

make_data_frame('./data') # build Random Dataset


#%%

data = pd.read_csv(DATA_PATH_LOGIT)

# 비율 A/B 테스트
OnesideLogisticTesting(data, group_col = 'group', target_col = 'is_purchase', alpha = 0.05)
TwosideLogisticTesting(data, group_col = 'group', target_col = 'is_purchase', alpha = 0.05)

data = pd.read_csv(DATA_PATH_NORM)

# A/B 테스트
OnesideTesting(data, group_col = 'group', target_col = 'purchase', alpha = 0.05)
TwosideTesting(data, group_col = 'group', target_col = 'purchase', alpha = 0.05)


data = pd.read_csv(DATA_PATH_GAMMA)

# A/B 테스트
OnesideTesting(data, group_col = 'group', target_col = 'purchase', alpha = 0.05)
TwosideTesting(data, group_col = 'group', target_col = 'purchase', alpha = 0.05)




    
#%%
# new line

#%%

# 여기 내용은 구현할것.
# https://doubly8f.netlify.app/%EA%B0%9C%EB%B0%9C/2020/08/11/ab-test-all/


# 결과 비교 화면은 이걸로 구현
# https://brunch.co.kr/@herbeauty/54

# 이건 뭘까?
# https://namofvietnam.medium.com/before-after-a-b-testing-statistics-for-marketing-analytics-with-both-r-and-python-25eed4453e2

# 로그형태 생각한 최종 구현
# 구조상 얘가 더 낫다.
# https://joshua-data.medium.com/harnessing-the-power-of-bigquery-and-python-overcoming-google-optimize-a-b-testing-limitations-9233365a0707


# 카이제곱 동질성 검정? 당근마켓 케이스
# 
# 평균에서 30배수 이상 값은 이상값
# 당근마켓에서는 평균으로부터 표준편차의 30배 이상 떨어지면 outlier라고 보고 값 제외 실시.
# 비교 기준은 count 지표와 value_sum