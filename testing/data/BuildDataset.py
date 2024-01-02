import numpy as np
import pandas as pd

class cfg:
    sample_size = 1000         # total size
    group_a_size = 0.48        # group size
    group_b_size = 0.52        # group size
    group_a_prob = [0.42,0.58] # logistic
    group_b_prob = [0.34,0.66] # logistic
    group_a_normal_params = {'mu' : 120, 'sigma' : 15} # N~(mu, sigma)
    group_b_normal_params = {'mu' : 100, 'sigma' : 12} # N~(mu, sigma)
    # https://homepage.stat.uiowa.edu/~mbognar/applets/gamma.html : 그래프 형태 참조.
    group_a_gamma_params = {'alpha' : 3.5, 'beta' : 35} # Gamma~(a, b)
    group_b_gamma_params = {'alpha' : 3, 'beta' : 35} # Gamma~(a, b)
    
    
def make_data_frame(path):
    GetLogisticDataset().to_csv(path+'/logistic_dataset.csv', index = False)
    GetNormalDataset().to_csv(path+'/normal_dataset.csv', index = False)
    GetGammaDataset().to_csv(path+'/gamma_dataset.csv', index = False)    
    
    return




def GetLogisticDataset():
    group = np.random.choice(['A','B'], size=cfg.sample_size, replace=True, p = [cfg.group_a_size, cfg.group_b_size])
    is_purchase = []

    for g in group:
        if g == 'A':
            is_purchase.append(np.random.choice([1,0], size=1, replace=True, p = cfg.group_a_prob)[0])
        else:
            is_purchase.append(np.random.choice([1,0], size=1, replace=True, p = cfg.group_b_prob)[0])

    return pd.DataFrame({'group' : group, 'is_purchase' : is_purchase})
    

def GetNormalDataset():
    group = np.random.choice(['A','B'], size=cfg.sample_size, replace=True, p = [cfg.group_a_size, cfg.group_b_size])
    purchase = []

    for g in group:
        if g == 'A':
            purchase.append(np.random.normal(cfg.group_a_normal_params['mu'], cfg.group_a_normal_params['sigma'], 1)[0])
        else:
            purchase.append(np.random.normal(cfg.group_b_normal_params['mu'], cfg.group_b_normal_params['sigma'], 1)[0])

    return pd.DataFrame({'group' : group, 'purchase' : purchase})


def GetGammaDataset():
    group = np.random.choice(['A','B'], size=cfg.sample_size, replace=True, p = [cfg.group_a_size, cfg.group_b_size])
    purchase = []

    for g in group:
        if g == 'A':
            purchase.append(np.random.gamma(cfg.group_a_gamma_params['alpha'], cfg.group_a_gamma_params['beta'], 1)[0])
        else:
            purchase.append(np.random.gamma(cfg.group_b_gamma_params['alpha'], cfg.group_b_gamma_params['beta'], 1)[0])

    return pd.DataFrame({'group' : group, 'purchase' : purchase})



