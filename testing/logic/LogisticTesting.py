import pandas as pd

import scipy.stats as stats
from scipy.stats import chi2_contingency
from statsmodels.stats import proportion

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns


# 이항분포의 정규성 근사 조건 확인 : np, npq가 5이상인 경우 CLT에 의해 정규성 근사
def is_NormalApproximation(na, nb, pa, pb):
    # group A, B의 np, npq가 5보다 큰 경우 CLT에 의해 이항분포의 정규성 근사
    if (na * pa < 5) or (na * (1 - pa) < 5) or (nb * pb < 5) or (nb * (1 - pb) < 5):
        return False # p가 작아서 발생하는 이슈기 때문에, 샘플 사이즈 늘려야
    else:
        return True
    
def OnesideLogisticTesting(data, group_col = 'group', target_col = 'is_purchase', alpha = 0.05):
    groupA_label, groupB_label = data.groupby(group_col)[target_col].count().index.tolist()[:2]
    na = data.groupby(group_col)[target_col].count()[0]
    nb = data.groupby(group_col)[target_col].count()[1]
    
    xa = data.groupby(group_col)[target_col].sum()[0] # boolean(0 or 1) sum
    xb = data.groupby(group_col)[target_col].sum()[1] # boolean(0 or 1) sum

    pa = xa / na
    pb = xb / nb
    
    # data rebalancing
    if pa > pb:
        rebalance_a = (groupB_label, nb, xb, pb)
        rebalance_b = (groupA_label, na, xa, pa)
        
        # A 확률이 B 확률보다 큰 경우, A그룹과 B 그룹의 변수명 변환
        groupA_label, na, xa, pa = rebalance_a
        groupB_label, nb, xb, pb = rebalance_b
        
    
    if (is_NormalApproximation(na, nb, pa, pb) == False):
        print('정규성 근사 불가, 더 많은 표본이 필요')
        return 
    
    z, p_value = proportion.proportions_ztest(
        count = [xa, xb],
        nobs = [na, nb],
        alternative = 'smaller' # two-sided, smaller, larger
    )
    
    print('단측 검정 T-Test (비율)')
    print('=====================================')
    print('설정된 유의 확률 : ', f'{str((1-alpha) * 100)}% (유의 수준 : {alpha})')
    
    if p_value < alpha:
      print(f'귀무 가설 기각, {groupA_label}안은 {groupB_label}안보다 작음 ({groupA_label} < {groupB_label})') 
    else:
      print(f'대립 가설 기각, {groupA_label}안은 {groupB_label}안보다 크거나 같음 ({groupA_label} >= {groupB_label})')
     
    (a_lower, b_lower), (a_upper, b_upper) = proportion.proportion_confint(
        [xa, xb],
        nobs = [na, nb],
        alpha = alpha
    )

    print('=====================================')
    print(f'p-value = {p_value:.4f}')
    print(f'> 그룹 {groupA_label}의 비율:', f'{pa * 100:.2f}%', f'({str((1-alpha) * 100)}% Confidence Interval: {a_lower * 100:.2f}% ~ {a_upper * 100:.2f}%)')
    print(f'> 그룹 {groupB_label}의 비율:', f'{pb * 100:.2f}%', f'({str((1-alpha) * 100)}% Confidence Interval: {b_lower * 100:.2f}% ~ {b_upper * 100:.2f}%)')    
    print('=====================================')
    
    
    # visualize
    plt.figure(figsize=(5,2.5))

    plt.plot(
        (a_lower*100, a_upper*100), (0, 0),  marker='o', color='blue'
    )
    plt.plot(
        (b_lower*100, b_upper*100), (1, 1),  marker='o', color='red'
    )
    plt.yticks(range(2), ['A', 'B'])
    plt.title(f'Conversion Rate ({str((1-alpha) * 100)}% Confidence Level)')
    plt.xlabel('Conversion Rate (%)')
    plt.ylabel('Group')

    plt.show();


        
def TwosideLogisticTesting(data, group_col = 'group', target_col = 'is_purchase', alpha = 0.05):
    groupA_label, groupB_label = data.groupby(group_col)[target_col].count().index.tolist()[:2]
    na = data.groupby(group_col)[target_col].count()[0]
    nb = data.groupby(group_col)[target_col].count()[1]
    
    xa = data.groupby(group_col)[target_col].sum()[0] # boolean(0 or 1) sum
    xb = data.groupby(group_col)[target_col].sum()[1] # boolean(0 or 1) sum

    pa = xa / na
    pb = xb / nb
    
    # data rebalancing
    if pa > pb:
        rebalance_a = (groupB_label, nb, xb, pb)
        rebalance_b = (groupA_label, na, xa, pa)
        
        # A 확률이 B 확률보다 큰 경우, A그룹과 B 그룹의 변수명 변환
        groupA_label, na, xa, pa = rebalance_a
        groupB_label, nb, xb, pb = rebalance_b
        
    
    if (is_NormalApproximation(na, nb, pa, pb) == False):
        print('정규성 근사 불가, 더 많은 표본이 필요')
        return 
    
    z, p_value = proportion.proportions_ztest(
        count = [xa, xb],
        nobs = [na, nb],
        alternative = 'two-sided' # two-sided, smaller, larger
    )
    
    print('양측 검정 T-Test (비율)')
    print('=====================================')
    print('설정된 유의 확률 : ', f'{str((1-alpha) * 100)}% (유의 수준 : {alpha})')
    
    if p_value < alpha:
      print(f'귀무 가설 기각, {groupA_label}안은 {groupB_label}안과 다름 ({groupA_label} != {groupB_label})') 
    else:
      print(f'대립 가설 기각, {groupA_label}안은 {groupB_label}안과 동일함 ({groupA_label} = {groupB_label})')
     
    (a_lower, b_lower), (a_upper, b_upper) = proportion.proportion_confint(
        [xa, xb],
        nobs = [na, nb],
        alpha = alpha
    )

    print('=====================================')
    print(f'p-value = {p_value:.4f}')
    print(f'> 그룹 {groupA_label}의 비율:', f'{pa * 100:.2f}%', f'({str((1-alpha) * 100)}% Confidence Interval: {a_lower * 100:.2f}% ~ {a_upper * 100:.2f}%)')
    print(f'> 그룹 {groupB_label}의 비율:', f'{pb * 100:.2f}%', f'({str((1-alpha) * 100)}% Confidence Interval: {b_lower * 100:.2f}% ~ {b_upper * 100:.2f}%)')    
    print('=====================================')
    
    
    # visualize
    plt.figure(figsize=(5,2.5))

    plt.plot(
        (a_lower*100, a_upper*100), (0, 0), marker='o', color='blue'
    )
    plt.plot(
        (b_lower*100, b_upper*100), (1, 1), marker='o', color='red'
    )
    plt.yticks(range(2), ['A', 'B'])
    plt.title(f'Conversion Rate ({str((1-alpha) * 100)}% Confidence Level)')
    plt.xlabel('Conversion Rate (%)')
    plt.ylabel('Group')

    plt.show();
