#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Song'

from path import Path
from pandas import *
import pandas as pd
import numpy as np


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


def get_3d_ordered_sample(uncensored, df):
    later_examples = []
    for row_index, row in uncensored.iterrows():
        later_example = df[df['Overall_Survival'] >= row['Overall_Survival']]
        later_example.pop('vital.status')
        later_example = np.asarray(later_example.drop(row_index))
        n_zeros_rows = len(df) - later_example.shape[0]
        zeros = np.zeros((n_zeros_rows, 295))
        later_example = np.concatenate([later_example, zeros])
        later_examples.append(later_example)
    return np.asarray(later_examples)


def get_at_risk_index(df):
    index = []
    for row_index, row in df.iterrows():
        later_example = df[df['Overall_Survival'] >= row['Overall_Survival']]
        # later_example = np.asarray(later_example.drop(row_index))
        n_zeros_rows = len(df) - later_example.shape[0]
        index.append(n_zeros_rows)
    # index[-1] = 0
    # print index
    return index


def load_data(dataset):
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
    df.insert(0, 'vital.status', observed)
    df.fillna(0, inplace=True)

    df = df.sort('Overall_Survival')   # sort based on survival time
    at_risk = get_at_risk_index(df[len(df)/3:])  # get the at risk for the later 2 / 3 training examples
    avg_series = DataFrame(avg_map, columns=df.columns)
    avg_series.pop('Overall_Survival')
    survival_time = df.pop('Overall_Survival')
    df.pop('vital.status')

    return np.asarray(observed, dtype='int32'), np.asarray(df), np.asarray(survival_time), np.asarray(at_risk)


if __name__ == '__main__':
    observed, X, survival_time, at_risk = load_data('C:/Users/Song/Research/biomed/Survival/trainingData.csv')