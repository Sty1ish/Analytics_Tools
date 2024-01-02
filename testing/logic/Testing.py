import numpy as np
import pandas as pd

# Testing
import scipy.stats as stats
from scipy.stats import chi2_contingency
from statsmodels.stats import proportion

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns


# sigma의 30배 초과하는 이상값 제거.
def drop_outlier(data, group_col = 'group', target_col = 'purchase', sigma_threshold = 30):
    # sigma_threshold : 기본값 30, 표준편차의 30배 이상 차이나는 값은 표본에서 제외
    # 참조 : https://medium.com/daangn/100-%ED%8C%80%EC%9B%90%EC%9D%98-%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EC%97%90-%EC%98%81%ED%96%A5%EC%9D%84-%EC%A3%BC%EB%8A%94-data-scientist-decision-5c939e8a3ea9
    tempdf = []
    
    for idx, valdf in data.groupby(group_col):
        max_val = valdf[target_col].mean() + (sigma_threshold * valdf[target_col].std())
        min_val = valdf[target_col].mean() - (sigma_threshold * valdf[target_col].std())
        
        tempdf.append(
            valdf.where(valdf[target_col] < max_val).where(valdf[target_col] > min_val).dropna()
            )
    
    return pd.concat(tempdf, axis = 0)


def is_NormalDist(groupA, groupB, alpha = 0.05):    
    shapiro_testA = stats.shapiro(groupA)
    shapiro_testB = stats.shapiro(groupB)
    
    if (shapiro_testA.pvalue >= alpha) and (shapiro_testB.pvalue >= alpha):
        return True
    else:
        # 귀무가설을 기각할 경우, 정규성 만족 X
        print('데이터가 정규성을 만족하지 않음')
        print(f'A group shapiro_test p-val : {shapiro_testA.pvalue}')
        print(f'B group shapiro_test p-val : {shapiro_testB.pvalue}')
        return False
    

def is_EqualVariance(groupA, groupB, alpha = 0.05):
    f, p_value = stats.levene(groupA, groupB)

    print("등분산 검정")
    print(f'Levene-Test p-value: {p_value:.3f}', end = '\t')
    
    if p_value > alpha:
        print("> 귀무가설 채택, 집단은 등분산성을 만족함")
        return True
    else:
        print("> 귀무가설 기각, 집단은 등분산성을 만족하지 않음")
        return False


# 신뢰구간 알고리즘이 확실하지 않음.
def OnesideTesting(dataframe, group_col = 'group', target_col = 'purchase', alpha = 0.05):
    # drop_outlier
    print('단측 검정 T-test')
    data = drop_outlier(dataframe, group_col = group_col, target_col = target_col, sigma_threshold = 30)
    print(f'전체 데이터 : {len(dataframe)}')
    print(f'이상값으로 제외된 데이터 : {len(dataframe) - len(data)}')
    
    # data setting
    groupA_label, groupB_label = data.groupby(group_col)[target_col].count().index.tolist()[:2]
    
    groupA = data[data[group_col] == groupA_label]
    groupB = data[data[group_col] == groupB_label]
    
    # data rebalancing
    if groupA[target_col].mean() > groupB[target_col].mean():
        groupA, groupB = groupB.copy(), groupA.copy()
        groupA_label, groupB_label = groupB_label, groupA_label
    
    
    # is_nomalDist
    if (is_NormalDist(groupA[target_col], groupB[target_col]) == False):
        print(f'[A group : {groupA_label}, B group : {groupB_label}]')
        print('=====================================')
        print(f'정규분포 X - Wilcox Test')
        wilcoxT, p_value = stats.wilcoxon(groupA[target_col]-groupB[target_col].mean(), alternative='less', zero_method='wilcox', correction=False)
        if (p_value < alpha):
            print(f'{groupA_label}안은 {groupB_label}안보다 작음 ({groupA_label} < {groupB_label})')
        else:
            
            print(f'A group : {groupA_label}, B group : {groupB_label}의 평균은 같다')
        
        print(f'p_value : {p_value}')
        print(f'> 그룹 {groupA_label}의 평균:', f'{groupA[target_col].mean():.2f}')
        print(f'> 그룹 {groupB_label}의 평균:', f'{groupB[target_col].mean():.2f}')    
        print('=====================================')
        
        return False
    
    print('=====================================')
    
    # is_EqualVariance
    if (is_EqualVariance(groupA[target_col], groupB[target_col]) == False):
        print('Welchs T-Test')
        t, p_value = stats.ttest_ind(groupA[target_col], groupB[target_col], equal_var = False, alternative = 'less')
        
        if p_value < alpha:
          print(f'귀무 가설 기각, {groupA_label}안은 {groupB_label}안보다 작음 ({groupA_label} < {groupB_label})') 
        else:
          print(f'대립 가설 기각, {groupA_label}안은 {groupB_label}안보다 크거나 같음 ({groupA_label} >= {groupB_label})')
        
        rv = stats.norm(0,1) 
        T_val = rv.ppf(1-(alpha*0.5)) # 단측이니까 /2 안해야 하지 않을까?
        
        cov = np.sqrt(((groupA[target_col].std() ** 2) / (groupA[target_col].count())) +  \
                      ((groupB[target_col].std() ** 2) / (groupB[target_col].count())))
              
        interval = T_val*cov
        
         
    else:
        print('Student T-Test')
        t, p_value = stats.ttest_ind(groupA[target_col], groupB[target_col], equal_var = True, alternative = 'less')
        
        if p_value < alpha:
          print(f'귀무 가설 기각, {groupA_label}안은 {groupB_label}안보다 작음 ({groupA_label} < {groupB_label})') 
        else:
          print(f'대립 가설 기각, {groupA_label}안은 {groupB_label}안보다 크거나 같음 ({groupA_label} >= {groupB_label})')
        
        
        rv = stats.norm(0,1) 
        T_val = rv.ppf(1-(alpha*0.5)) # 단측이니까 /2 안해야 하지 않을까?
        
        cov = (((groupA[target_col].count() - 1)* (groupA[target_col].std() ** 2)) +  \
               ((groupB[target_col].count() - 1)* (groupB[target_col].std() ** 2)) ) / \
              (groupA[target_col].count() + groupB[target_col].count() - 2)
        
        interval = T_val*np.sqrt(cov*((1/groupA[target_col].count()) + (1/groupB[target_col].count())))
              
    
    print('=====================================')
    print(f'p-value = {p_value:.4f}')
    print(f'> 그룹 {groupA_label}의 평균:', f'{groupA[target_col].mean():.2f}', f'({str((1-alpha) * 100)}% Confidence Interval: {groupA[target_col].mean() - interval} ~ {groupA[target_col].mean() + interval}')
    print(f'> 그룹 {groupB_label}의 평균:', f'{groupB[target_col].mean():.2f}', f'({str((1-alpha) * 100)}% Confidence Interval: {groupB[target_col].mean() - interval} ~ {groupB[target_col].mean() + interval}')    
    print('=====================================')
    
    # 종별 분포
    fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    ax = sns.kdeplot(x=target_col, data=data, hue=group_col, fill=True, ax=ax)

    ax.axvline(groupA[target_col].mean(), linestyle = ':', color = 'black')
    ax.axvline(groupB[target_col].mean(), linestyle = ':', color = 'red')
    ax.text(x=groupA[target_col].mean(), y=0, s=round(groupA[target_col].mean(),1), color= 'black');
    ax.text(x=groupB[target_col].mean(), y=0, s=round(groupB[target_col].mean(),1), color= 'red');
    


    
    
# 신뢰구간 알고리즘이 확실하지 않음.
def TwosideTesting(dataframe, group_col = 'group', target_col = 'purchase', alpha = 0.05):
    # drop_outlier
    print('양측 검정 T-test')
    data = drop_outlier(dataframe, group_col = group_col, target_col = target_col, sigma_threshold = 30)
    print(f'전체 데이터 : {len(dataframe)}')
    print(f'이상값으로 제외된 데이터 : {len(dataframe) - len(data)}')
    
    # data setting
    groupA_label, groupB_label = data.groupby(group_col)[target_col].count().index.tolist()[:2]
    
    groupA = data[data[group_col] == groupA_label]
    groupB = data[data[group_col] == groupB_label]
    
    # data rebalancing
    if groupA[target_col].mean() > groupB[target_col].mean():
        groupA, groupB = groupB.copy(), groupA.copy()
        groupA_label, groupB_label = groupB_label, groupA_label
    
    
    # is_nomalDist
    if (is_NormalDist(groupA[target_col], groupB[target_col]) == False):
        print(f'[A group : {groupA_label}, B group : {groupB_label}]')
        print('=====================================')
        print(f'정규분포 X - Wilcox Test')
        wilcoxT, p_value = stats.wilcoxon(groupA[target_col]-groupB[target_col].mean(), alternative='two-sided', zero_method='wilcox', correction=False)
        if (p_value < alpha):
            print(f'{groupA_label}안은 {groupB_label}안보다 작음 ({groupA_label} < {groupB_label})')
        else:
            print(f'A group : {groupA_label}, B group : {groupB_label}의 평균은 같다')
        
        print(f'> 그룹 {groupA_label}의 평균:', f'{groupA[target_col].mean():.2f}')
        print(f'> 그룹 {groupB_label}의 평균:', f'{groupB[target_col].mean():.2f}')    
        print('=====================================')
        
        return False
    
    print('=====================================')
    
    # is_EqualVariance
    if (is_EqualVariance(groupA[target_col], groupB[target_col]) == False):
        print('Welchs T-Test')
        t, p_value = stats.ttest_ind(groupA[target_col], groupB[target_col], equal_var = False, alternative = 'two-sided')
        
        if p_value < alpha:
          print(f'귀무 가설 기각, {groupA_label}안은 {groupB_label}안과 다름 ({groupA_label} != {groupB_label})') 
        else:
          print(f'대립 가설 기각, {groupA_label}안은 {groupB_label}안과 같음 ({groupA_label} = {groupB_label})')
        
        rv = stats.norm(0,1) 
        T_val = rv.ppf(1-(alpha*0.5))
        
        cov = np.sqrt(((groupA[target_col].std() ** 2) / (groupA[target_col].count())) +  \
                      ((groupB[target_col].std() ** 2) / (groupB[target_col].count())))
              
        interval = T_val*cov
        
         
    else:
        print('Student T-Test')
        t, p_value = stats.ttest_ind(groupA[target_col], groupB[target_col], equal_var = True, alternative = 'two-sided')
        
        if p_value < alpha:
          print(f'귀무 가설 기각, {groupA_label}안은 {groupB_label}안과 다름 ({groupA_label} != {groupB_label})') 
        else:
          print(f'대립 가설 기각, {groupA_label}안은 {groupB_label}안과 같음 ({groupA_label} = {groupB_label})')
        
        
        rv = stats.norm(0,1) 
        T_val = rv.ppf(1-(alpha*0.5))
        
        cov = (((groupA[target_col].count() - 1)* (groupA[target_col].std() ** 2)) +  \
               ((groupB[target_col].count() - 1)* (groupB[target_col].std() ** 2)) ) / \
              (groupA[target_col].count() + groupB[target_col].count() - 2)
        
        interval = T_val*np.sqrt(cov*((1/groupA[target_col].count()) + (1/groupB[target_col].count())))
              
    
    print('=====================================')
    print(f'p-value = {p_value:.4f}')
    print(f'> 그룹 {groupA_label}의 평균:', f'{groupA[target_col].mean():.2f}', f'({str((1-alpha) * 100)}% Confidence Interval: {groupA[target_col].mean() - interval} ~ {groupA[target_col].mean() + interval}')
    print(f'> 그룹 {groupB_label}의 평균:', f'{groupB[target_col].mean():.2f}', f'({str((1-alpha) * 100)}% Confidence Interval: {groupB[target_col].mean() - interval} ~ {groupB[target_col].mean() + interval}')    
    print('=====================================')
        
    # 종별 분포
    fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    ax = sns.kdeplot(x=target_col, data=data, hue=group_col, fill=True, ax=ax)

    ax.axvline(groupA[target_col].mean(), linestyle = ':', color = 'black')
    ax.axvline(groupB[target_col].mean(), linestyle = ':', color = 'red')
    ax.text(x=groupA[target_col].mean(), y=0, s=round(groupA[target_col].mean(),1), color= 'black');
    ax.text(x=groupB[target_col].mean(), y=0, s=round(groupB[target_col].mean(),1), color= 'red');
      
        