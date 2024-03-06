import numpy as np, scipy.io, os, multiprocessing
import itertools
import pandas as pd
import warnings

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, KFold

from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


def model_list(grid=False, seed=123):
    
    xgbr_param = {'nthread':[4], 'objective':['reg:squarederror'], 'learning_rate': [.03, 0.05, .07], 'max_depth': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 'min_child_weight': [4], 
                                     'subsample': [0.7], 'colsample_bytree': [0.7], 'n_estimators': [400, 450, 500, 550, 600, 660]}
    XGBR_ =XGBRegressor(random_state = seed)



    lgbmr_param =  {'num_leaves': [7, 14, 21], 'learning_rate': [0.01, 0.05, 0.001, 0.005], 'max_depth': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 
                                        'min_data_in_leaf':[10, 15, 25], 'feature_fraction': [0.6, 0.8, 0.9],'cat_smooth': [1,10, 15, 20, 35], 'verbose': [-1]}  
    LGBMR_ = LGBMRegressor(random_state=seed, verbose=-1)


    CATBOOST_ = CatBoostRegressor(random_state=seed,loss_function='RMSE',eval_metric='RMSE')
    catboost_param = {'iterations':[100,150,200],'learning_rate':[0.03,0.1],'depth':[2,4,6,8],'l2_leaf_reg':[0.2,0.5,1,3]}



    if grid ==True:
        model_stack = { "XBGR":[XGBR_, xgbr_param], "LGBMR": [LGBMR_, lgbmr_param],"CATBOOST":[CATBOOST_,catboost_param]}

    else:
        model_stack = { "XBGR":XGBR_, "LGBMR": LGBMR_}

    return model_stack