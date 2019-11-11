"""model"""
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from util import log
from preprocess import sample
from additional_util import *

# dependencies for time and memory test
import psutil
import os
import time


class AutoSSLClassifier:
    def __init__(self):
        self.iter = 5
        self.label_data = None
        self.model = None
        self.max_sample_num = 100000
        self.unbalance_ratio = 0.75
        self.sample_ratio = 3
        self.unlabel_sample_ratio = 10


    def downsample(self, X, y, seed=1, pos=1, neg=-1):
        # find majority class
        positive_ratio = len(y[y==1]) / len(y)
        if (positive_ratio >= 0.5):
            majority_ratio = positive_ratio
            majority_class, minority_class = pos, neg
        else:
            majority_ratio = 1. - positive_ratio
            majority_class, minority_class = neg, pos
        # subsample based on sample number and majority ratio
        sample_num = y.shape[0]
        minority_sample_num = (y == minority_class).sum()
        if (sample_num > self.max_sample_num):
            if (majority_ratio > self.unbalance_ratio):
                if (minority_sample_num <= self.max_sample_num * (1. - self.unbalance_ratio)):
                    minority_index = y[y == minority_class].index
                else:
                    minority_index = y[y == minority_class].sample(int(self.max_sample_num * (1. - self.unbalance_ratio)), random_state=seed).index
                majority_index = y[y == majority_class].sample(int(minority_index.shape[0] * self.sample_ratio), random_state=seed).index
                sample_index = np.append(minority_index, majority_index)
            else:
                sample_index = y.sample(self.max_sample_num, random_state=seed).index
            X_sample, y_sample = X.loc[sample_index, :], y.loc[sample_index]
        else:
            if (majority_ratio > self.unbalance_ratio):
                minority_index = y[y == minority_class].index
                majority_index = y[y == majority_class].sample(int(minority_index.shape[0] * self.sample_ratio), random_state=seed).index
                sample_index = np.append(minority_index, majority_index)
                X_sample, y_sample = X.loc[sample_index, :], y.loc[sample_index]
            else:
                X_sample, y_sample = X, y
        return X_sample, y_sample


    def fit(self, X, y):

        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }


        X_label, y_label, X_unlabeled, y_unlabeled = self._split_by_label(X, y)
        value_num = y_label.value_counts()
        y_n_cnt, y_p_cnt = value_num[-1], value_num[1]

        y_n_ratio = y_n_cnt*1.0/len(y_label)
        y_p_ratio = y_p_cnt*1.0/len(y_label)

        # sample self.label_data every iter
        self.label_data = int(X_label.shape[0] * 4.0 / self.iter)

        X_rlabel, y_rlabel = self.downsample(X_label, y_label)
        self.model, hyper = train_gbm_model(X_rlabel, y_rlabel, params, weight=None, tune=True)

        for iters in range(self.iter):
            num_to_label = min(self.label_data, X_unlabeled.shape[0])
            y_n = max(int(num_to_label * y_n_ratio), 1)
            y_p = max(int(num_to_label * y_p_ratio), 1)

            if X_unlabeled.shape[0] == 0:
                break

            unlabel_label_ratio = X_unlabeled.shape[0] / X_label.shape[0]
            if (unlabel_label_ratio > self.unlabel_sample_ratio):
                pred_data = X_unlabeled.sample(num_to_label * self.unlabel_sample_ratio, random_state=0)
                pred_idx = pred_data.index
                y_hat = self.model.predict(pred_data)
            else:
                pred_idx = X_unlabeled.index
                y_hat = self.model.predict(X_unlabeled)
            out = pd.Series(y_hat, index=pred_idx)

            out = out.sort_values()
            # get the sorted idx
            sortedIdx = out.index

            if(X_unlabeled.shape[0] == 0):
                break
            log(f"{max(y_hat)}, {min(y_hat)}")
            y_p_com = y_p
            y_n_com = y_n
            y_p_idx = sortedIdx[-y_p_com:]
            y_n_idx = sortedIdx[:y_n_com]
            new_label_idx = y_p_idx.copy()
            new_label_idx.append(y_n_idx)
            X_label = pd.concat([X_label, X_unlabeled.loc[new_label_idx, :]])
            y_unlabeled[y_n_idx] = -1
            y_unlabeled[y_p_idx] = 1
            y_label = pd.concat([y_label, y_unlabeled[new_label_idx]])
            X_unlabeled.drop(index=new_label_idx, inplace=True)
            X_rlabel, y_rlabel = self.downsample(X_label, y_label)
            self.model, _ = train_gbm_model(X_rlabel, y_rlabel, params, weight=None, tune=False, new_params=hyper)

        return self


    def predict(self, X, pred_time_budget=None, remaining_time=None):

        pred_start_time = time.time()
        self.X_test = X.values.astype('float32', copy=False)

        if (self.model.num_trees() <= 400):
            return self.model.predict(self.X_test)

        unit_start_time = time.time()
        self.model.predict(self.X_test, num_iteration=50)
        unit_end_time = time.time()
        unit_time = unit_end_time - unit_start_time

        max_iteration = int(50 * 0.9 * (remaining_time - (unit_end_time - pred_start_time)) / unit_time)
        iteration = min(max_iteration, self.model.num_trees())
        return self.model.predict(self.X_test, num_iteration=iteration)


    def _split_by_label(self, X, y):
        y_label = y[y != 0]
        X_label = X.loc[y_label.index, :]
        y_unlabeled = y[y == 0]
        X_unlabeled = X.loc[y_unlabeled.index, :]
        return X_label, y_label, X_unlabeled, y_unlabeled


class AutoPUClassifier:
    def __init__(self):
        self.iter = 5
        self.models = []
        self.max_sample_num = 100000
        self.unbalance_ratio = 0.75
        self.sample_ratio = 3


    def downsample(self, X, y, seed=1):
        # find majority class
        positive_ratio = y.mean()
        if (positive_ratio >= 0.5):
            majority_ratio = positive_ratio
            majority_class, minority_class = 1, 0
        else:
            majority_ratio = 1. - positive_ratio
            majority_class, minority_class = 0, 1
        # subsample based on sample number and majority ratio
        sample_num = y.shape[0]
        minority_sample_num = (y == minority_class).sum()
        if (sample_num > self.max_sample_num):
            if (majority_ratio > self.unbalance_ratio):
                if (minority_sample_num <= self.max_sample_num * (1. - self.unbalance_ratio)):
                    minority_index = y[y == minority_class].index
                else:
                    minority_index = y[y == minority_class].sample(int(self.max_sample_num * (1. - self.unbalance_ratio)), random_state=seed).index
                majority_index = y[y == majority_class].sample(int(minority_index.shape[0] * self.sample_ratio), random_state=seed).index
                sample_index = np.append(minority_index, majority_index)
            else:
                np.random.seed(seed)
                sample_index = np.random.choice(sample_num, size=self.max_sample_num, replace=False)
            X_sample, y_sample = X.loc[sample_index, :], y.loc[sample_index]
        else:
            if (majority_ratio > self.unbalance_ratio):
                minority_index = y[y == minority_class].index
                majority_index = y[y == majority_class].sample(int(minority_index.shape[0] * self.sample_ratio), random_state=seed).index
                sample_index = np.append(minority_index, majority_index)
                X_sample, y_sample = X.loc[sample_index, :], y.loc[sample_index]
            else:
                X_sample, y_sample = X, y
        return X_sample, y_sample


    def fit(self, X, y):
        share_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1, 
            "num_threads": 4
        }
        gbm_params = {
            "learning_rate": 0.01, 
            'num_iterations': 2000, 
            'max_depth': 6, 
            "num_leaves": 2 ** 6,
            'bagging_fraction': 0.8, 
            'bagging_freq': 1, 
            "feature_fraction": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_samples": 5
        }

        # bagging of LGB with different majority downsampling
        for i in range(1, self.iter + 1):
            X_sample, y_sample = self.downsample(X, y, seed=i)
            # do HPO only on the first model
            if (i == 1):
                gbm_params = HPO_gbm(X_sample, y_sample, share_params, second_level_tune=True)
            X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.1, random_state=i, stratify=y_sample)
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train({**share_params, **gbm_params}, train_data, valid_sets=valid_data, 
                early_stopping_rounds=50, verbose_eval=100)
            self.models.append(model)

        return self


    def predict(self, X, pred_time_budget=None, remaining_time=None):

        pred_start_time = time.time()

        if (isinstance(X, pd.DataFrame)):
            self.X_test = X.values.astype('float32', copy=False)
        else:
            self.X_test = X

        preds = []
        init_remaining_time = remaining_time
        min_remaining_time = pred_time_budget * 0.2

        # start predicting from the model with the smallest iterations
        tree_num = np.array([model.num_trees() for model in self.models])
        model_order = np.argsort(tree_num)
        max_tree_num = tree_num.max()

        # intial estimation of unit predict time
        unit_tree_num = 200
        unit_tree_num = min(unit_tree_num, max_tree_num)
        unit_start_time = time.time()
        self.models[model_order[-1]].predict(self.X_test, num_iteration=unit_tree_num)
        unit_end_time = time.time()
        unit_time = unit_end_time - unit_start_time
        print ('unit time: %.2f' %(unit_time))

        remaining_time = init_remaining_time - (time.time() - pred_start_time)

        for i, idx in enumerate(model_order):

            iter_start_time = time.time()
            
            estimated_pred_time = unit_time * tree_num[idx] / unit_tree_num
            # use full model to predict if the time budget is enough
            if (estimated_pred_time < (remaining_time - min_remaining_time)):
                preds.append(self.models[idx].predict(self.X_test))
                unit_tree_num = tree_num[idx]
                print ('make full prediction on model %d' %(i))
            # early stop the current model and discard the rest models if the budget is not enough
            else:
                if ((remaining_time - min_remaining_time) <= 0):
                    estimated_max_iteration = 10
                else:
                    estimated_max_iteration = max(int(unit_tree_num * (remaining_time - min_remaining_time) / unit_time), 10)
                early_stop_iteration = min(estimated_max_iteration, tree_num[idx])
                preds.append(self.models[idx].predict(self.X_test, num_iteration=early_stop_iteration))
                unit_tree_num = early_stop_iteration
                print ('early stop at round %d on model %d' %(early_stop_iteration, i))

            iter_end_time = time.time()

            # update unit predict time
            print ('time spent on model %d: %.2f' %(i, iter_end_time - iter_start_time))
            unit_time = iter_end_time - iter_start_time

            remaining_time = init_remaining_time - (time.time() - pred_start_time)
            if ((remaining_time / init_remaining_time <= 0.1) or (remaining_time <= min_remaining_time)):
                print ('discard the models after iter %d' %(i))
                break

        self.y_pred = np.vstack(preds).mean(axis=0)
        return self.y_pred


class AutoNoisyClassifier:
    def __init__(self):
        self.models = []
        self.max_sample_num = 100000
        self.unbalance_ratio = 0.75
        self.sample_ratio = 3


    def fit(self, X, y):
        share_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }
        rf_params = {
            'boosting': 'rf', 
            'num_iterations': 200, 
            'max_depth': 6, 
            'num_leaves': 2 ** 6, 
            'min_child_samples': 5, 
            'bagging_fraction': 0.8, 
            'bagging_freq': 1, 
            'feature_fraction': 1. / np.sqrt(X.shape[1]), 
            'max_bin': 255
        }

        # preprocess
        y[y == -1] = 0
        # find majority class
        positive_ratio = y.mean()
        if (positive_ratio >= 0.5):
            majority_ratio = positive_ratio
            majority_class, minority_class = 1, 0
        else:
            majority_ratio = 1. - positive_ratio
            majority_class, minority_class = 0, 1
        # subsample based on sample number and majority ratio
        sample_num = y.shape[0]
        minority_sample_num = (y == minority_class).sum()
        if (sample_num > self.max_sample_num):
            if (majority_ratio > self.unbalance_ratio):
                if (minority_sample_num <= self.max_sample_num * (1. - self.unbalance_ratio)):
                    minority_index = y[y == minority_class].index
                else:
                    minority_index = y[y == minority_class].sample(int(self.max_sample_num * (1. - self.unbalance_ratio)), random_state=1).index
                majority_index = y[y == majority_class].sample(int(minority_index.shape[0] * self.sample_ratio), random_state=1).index
                sample_index = np.append(minority_index, majority_index)
            else:
                np.random.seed(1)
                sample_index = np.random.choice(sample_num, size=self.max_sample_num, replace=False)
            X, y = X.loc[sample_index, :], y.loc[sample_index]
        else:
            if (majority_ratio > self.unbalance_ratio):
                minority_index = y[y == minority_class].index
                majority_index = y[y == majority_class].sample(int(minority_index.shape[0] * self.sample_ratio), random_state=1).index
                sample_index = np.append(minority_index, majority_index)
                X, y = X.loc[sample_index, :], y.loc[sample_index]

        # filter noisy samples with directly training scores
        threshold = 0.5
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, 
            max_features='auto', bootstrap=True, n_jobs=4, random_state=1)
        clf.fit(X, y)
        y_pred = clf.predict_proba(X)[:, 1]

        d = np.abs(y_pred - y.values)
        print ('percentile of the difference of y_pred and y')
        print (np.percentile(d, list(range(0, 110, 10))))

        valid_threshold = np.percentile(d, 10.)
        print ('valid threshold: %.4f' %(valid_threshold))
        valid_index = (d < valid_threshold)
        X_valid, y_valid = X.iloc[valid_index, :], y.iloc[valid_index]

        if (np.unique(y_valid).shape[0] == 1):
            score_1, score_2 = 1., 1.
            print ('only one class in validation set')
        else:
            # method 1: remove noisy samples
            train_index = (d >= valid_threshold) & (d <= threshold)
            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
            train_data = lgb.Dataset(X_train, label=y_train)
            model_1 = lgb.train({**share_params, **rf_params}, train_data, verbose_eval=100)
            score_1 = roc_auc_score(y_valid.values, model_1.predict(X_valid))
            print ('method 1: %.4f' %(score_1))
            # method 2: weight noisy samples by d
            train_index = (d >= valid_threshold)
            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
            weight = np.ones(y_train.shape[0], dtype=float)
            noise_index = (d[train_index] > threshold)
            weight[noise_index] = 1. - d[train_index][noise_index]
            train_data = lgb.Dataset(X_train, label=y_train, weight=weight)
            model_2 = lgb.train({**share_params, **rf_params}, train_data, verbose_eval=100)
            score_2 = roc_auc_score(y_valid.values, model_2.predict(X_valid))
            print ('method 2: %.4f' %(score_2))

        epsilon = 0.005
        if (score_1 - score_2 > epsilon):
            train_index = (d <= threshold)
            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
            train_data = lgb.Dataset(X_train, label=y_train)
            model_1 = lgb.train({**share_params, **rf_params}, train_data, verbose_eval=100)
            self.models.append(model_1)
            print ('remove noisy samples')
        elif (score_1 - score_2 < -epsilon):
            weight = np.ones(y.shape[0], dtype=float)
            noise_index = (d > threshold)
            weight[noise_index] = (1. - d)[noise_index]
            train_data = lgb.Dataset(X, label=y, weight=weight)
            model_2 = lgb.train({**share_params, **rf_params}, train_data, verbose_eval=100)
            self.models.append(model_2)
            print ('weight noisy samples')
        else:
            train_index = (d <= threshold)
            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
            train_data = lgb.Dataset(X_train, label=y_train)
            model_1 = lgb.train({**share_params, **rf_params}, train_data, verbose_eval=100)
            self.models.append(model_1)
            
            weight = np.ones(y.shape[0], dtype=float)
            noise_index = (d > threshold)
            weight[noise_index] = (1. - d)[noise_index]
            train_data = lgb.Dataset(X, label=y, weight=weight)
            model_2 = lgb.train({**share_params, **rf_params}, train_data, verbose_eval=100)
            self.models.append(model_2)
            print ('use both methods')

        return self


    def predict(self, X, pred_time_budget=None, remaining_time=None):
        if (isinstance(X, pd.DataFrame)):
            self.X_test = X.values.astype('float32', copy=False)
        else:
            self.X_test = X
        preds = [model.predict(self.X_test) for model in self.models]
        return np.vstack(preds).mean(axis=0)