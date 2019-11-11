import datetime

import CONSTANT
from util import log, timeit
from collections import Counter

import numpy as np
import pandas as pd
import gc
import psutil
import os


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


def encode(series, freq=None):
    if freq is None:
        count = pd.Series(Counter(series))
        freq = (count / series.shape[0]).astype('float32')
    series = series.map(freq)
    series.fillna(0., inplace=True)
    return series, freq


@timeit
def transform_categorical_freq(df, model, cat_cols):
    for c in cat_cols:
        if model.training:
            df[c + '_Afreq'], freq = encode(df[c])
            model.param_to_save[c + '_Afreq'] = freq
        else:
            df[c + '_Afreq'], _ = encode(df[c], model.param_to_save[c + '_Afreq'])
    return pd.Index([c + '_Afreq' for c in cat_cols])


@timeit
def transform_numerical_type(df, num_cols):
    for c in num_cols:
        df[c] = df[c].astype('float32', copy=False)
    gc.collect()


@timeit
def feature_engineer(df, model):
    gc.collect()
    num_cols = pd.Index(model.dtype_cols['num'])
    cat_cols = pd.Index(model.dtype_cols['cat'])
    addi_freq = transform_categorical_freq(df, model, cat_cols)
    addi_lbe = transform_categorical_cat2num(df, model, cat_cols, inplace=False)
    used_cols = num_cols.append(addi_freq).append(addi_lbe)
    drop_feature(df, df.keys().difference(used_cols))

        
@timeit
def drop_feature(df, dropList):
    df.drop(dropList, axis=1, inplace=True)


@timeit
def transform_categorical_cat2num(df, model, cat_cols, inplace=False):
    added_col = []
    for c in cat_cols:
        if model.training:
            cat = pd.Categorical(df[c])
            model.param_to_save[c + '_cat2num'] = cat.categories
            if inplace:
                df[c] = cat.codes
            else:
                df[c + '_cat2num'] = cat.codes
        else:
            if inplace:
                df[c] = pd.Categorical(df[c], categories=model.param_to_save[c + '_cat2num']).codes
            else:
                df[c + '_cat2num'] = pd.Categorical(df[c], categories=model.param_to_save[c + '_cat2num']).codes
        added_col.append(c if inplace else c+'_cat2num')
    return pd.Index(added_col)


@timeit
def sample(X, y, nrows):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample