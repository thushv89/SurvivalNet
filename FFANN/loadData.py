#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Song'

from path import Path
from pandas import *
import pandas as pd
import numpy as np
import theano


drop_list = ['Remission_Duration', 'resp.simple', 'Relapse']

categorical_features = [
                          'SEX',
                          'Chemo.Simplest',
                          'PRIOR.MAL',  # Whether the patient has previous cancer
                          'PRIOR.CHEMO',  # Whether the patient had prior chemo
                          'PRIOR.XRT',  # Prior radiation
                          'Infection',  # Has infection
                          'cyto.cat',  # cytogenic category
                          'ITD',  # Has the ITD FLT3 mutation
                          'D835',  # Has the D835 FLT3 mutation
                          'Ras.Stat'  # Has the Ras.Stat mutation
]

def fill_na_with_mean(column):
    avg = column.mean()
    column.fillna(avg, inplace=True)
    return avg

def load_training_data(dataset):
    print 'loading data'
    p = Path(dataset)
    df = pd.read_csv(p, index_col=0)
    avg_map = {}
    for key in drop_list:
        df.pop(key)
    observed = df.pop('vital.status')
    for column in categorical_features:
        groups = df.groupby(column).groups
        df.pop(column)
        for key in groups:
            name = column + '_' + key
            value = groups[key]
            new_column = Series(np.ones(len(value)), index=value)
            df.insert(0, name, new_column)
            avg_map[name] = [1]
    numerical_features = set(df.columns).difference(set(categorical_features))
    for column in numerical_features:
        avg = fill_na_with_mean(df[column])
        avg_map[column] = [avg]
    t_column = df.pop('Overall_Survival')
    df.insert(0, 'Overall_Survival', t_column)
    df.fillna(0, inplace=True)

    test_series = DataFrame(avg_map, columns=df.columns)
    test_series.pop('Overall_Survival')

    shared_x = theano.shared(np.asarray(df, dtype=theano.config.floatX), borrow=True)
    shared_t = theano.shared(np.asarray(t_column, dtype=theano.config.floatX), borrow=True)

    observed.replace(to_replace='A', value=0, inplace=True)
    observed.replace(to_replace='D', value=1, inplace=True)

    return shared_x, shared_t, t_column, observed, test_series

# load_training_data('C:/Users/Song/Research/biomed/trainingData.csv')
