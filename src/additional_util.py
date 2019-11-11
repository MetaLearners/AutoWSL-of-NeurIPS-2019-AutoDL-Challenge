import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from preprocess import sample
from util import log
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def train_gbm_model(X, y, params, tune=False, random_state=0, weight=None, new_params=None):
    if tune:
        hyper_params = HPO_gbm(X, y, params)
    else:
        if (new_params != None):
            hyper_params = new_params
        else:
            hyper_params = {
                'learning_rate': 0.01, 
                'num_iterations': 2000, 
                'max_depth': 6, 
                'num_leaves': 64, 
                "min_child_samples": 5, 
                'bagging_fraction': 0.8, 
                'bagging_freq': 1, 
                "feature_fraction": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1
            }
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state, stratify=y)

    if weight is None:
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
    else:
        train_data = lgb.Dataset(X_train, label=y_train, weight=get_weight_from_label(y_train, weight))
        valid_data = lgb.Dataset(X_val, label=y_val, weight=get_weight_from_label(y_val, weight))

    return lgb.train({**params, **hyper_params}, train_data, valid_sets=valid_data, 
        early_stopping_rounds=50, verbose_eval=100), hyper_params


def HPO_gbm(X, y, params, second_level_tune=True):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    # determine learning rate
    default_params = {
        'max_depth': 6, 
        'num_leaves': 64, 
        "min_child_samples": 5, 
        'bagging_fraction': 0.8, 
        'bagging_freq': 1, 
        "feature_fraction": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1
    }
    
    learning_rate = [0.002, 0.005, 0.01, 0.02, 0.05]
    valid_auc = []
    ES_iteration = []
    for lr in learning_rate:
        model = lgb.train({**params, **default_params, 'learning_rate': lr}, train_data, 2000, valid_data, 
            early_stopping_rounds=50, verbose_eval=100)
        ES_iteration.append(model.best_iteration)
        valid_auc.append(model.best_score["valid_0"][params["metric"]])
    print (learning_rate)
    print (valid_auc)
    print (ES_iteration)
    index = np.argmax(np.array(valid_auc))
    best_lr = learning_rate[index]
    best_iteration = ES_iteration[index]
    best_iteration = 200 * (int(best_iteration / 200) + 1)

    default_params['learning_rate'] = best_lr

    if not second_level_tune:
        default_params['num_iterations'] = best_iteration
        print ('best params found')
        print (default_params)
        return default_params

    max_depth_options = [3, 4, 5, 6, 7]
    min_child_samples_options = [1, 5, 10]
    scores = []
    models = []
    new_params = []

    # use grid search to do HPO
    # use low fidelity HPO if the iteration is large
    if (best_iteration > 1500):
        tune_iteration = int(best_iteration / 2)
    else:
        tune_iteration = best_iteration

    for max_depth in max_depth_options:
        for min_child_samples in min_child_samples_options:
            tune_params = default_params.copy()
            tune_params['max_depth'] = max_depth
            tune_params['num_leaves'] = 2 ** max_depth
            tune_params['min_child_samples'] = min_child_samples
            model = lgb.train({**params, **tune_params}, train_data, valid_sets=valid_data, num_boost_round=tune_iteration, 
                early_stopping_rounds=50, verbose_eval=100)
            # compute score with regularization
            score = model.best_score["valid_0"][params["metric"]] - 0.00075 * max_depth + 0.00005 * min_child_samples
            scores.append(score)
            models.append(model)
            new_params.append(tune_params)
    index = np.argmax(np.array(scores))
    best_params = new_params[index]
    best_params['num_iterations'] = best_iteration
    print ('best params found')
    print (best_params)
    return best_params