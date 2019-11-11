"""baseline"""
import os
import time
import pickle
import pandas as pd
from util import log, timeit
from automl import AutoSSLClassifier, AutoNoisyClassifier, AutoPUClassifier
from preprocess import fillna, feature_engineer


class Model:
    """entry"""
    def __init__(self, info: dict):
        log(f"Info:\n {info}")

        self.model = None
        self.task = info['task']
        self.train_time_budget = info['time_budget']
        self.pred_time_budget = info['pred_time_budget']
        self.cols_dtype = info['schema']
        self.info = info

        self.dtype_cols = {'cat': [], 'num': [], 'time': []}

        for key, value in self.cols_dtype.items():
            if value == 'cat':
                self.dtype_cols['cat'].append(key)
            elif value == 'num':
                self.dtype_cols['num'].append(key)
            elif value == 'time':
                self.dtype_cols['time'].append(key)

        self.training = True
        self.param_to_save = {}


    @timeit
    def train(self, X: pd.DataFrame, y: pd.Series):
        """train model"""
        start_time = time.time()

        fillna(X)
        feature_engineer(X, self)

        log(f"Remain time: {self.train_time_budget - (time.time() - start_time)}")

        if self.task == 'ssl':
            self.model = AutoSSLClassifier()
        elif self.task == 'pu':
            self.model = AutoPUClassifier()
        elif self.task == 'noisy':
            self.model = AutoNoisyClassifier()

        self.model.fit(X, y)


    @timeit
    def predict(self, X: pd.DataFrame):
        """predict"""
        self.pred_func_start_time = time.time()
        print ('time spent on loading model and test data: %.2f' %(self.pred_func_start_time - self.pred_start_time))

        self.training = False
        print ('test data size: %d' %(X.shape[0]))
            
        fillna(X)
        feature_engineer(X, self)
        
        remaining_time = self.pred_time_budget - (time.time() - self.pred_start_time)
        prediction = self.model.predict(X, pred_time_budget=self.pred_time_budget, remaining_time=remaining_time)

        return pd.Series(prediction)


    @timeit
    def save(self, directory: str):
        """save model"""
        pickle.dump(
            self.model, open(os.path.join(directory, 'model.pkl'), 'wb'))
        pickle.dump(
            self.param_to_save, open(os.path.join(directory, 'param.pkl'), 'wb'))


    @timeit
    def load(self, directory: str):
        self.pred_start_time = time.time()
        """load model"""
        self.model = pickle.load(
            open(os.path.join(directory, 'model.pkl'), 'rb'))
        self.param_to_save = pickle.load(
            open(os.path.join(directory, 'param.pkl'), 'rb'))
