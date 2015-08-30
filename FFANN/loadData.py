#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Song'

from path import Path
from pandas import *
from lifelines.coxph_fitter import CoxPHFitter
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


def discrete_time_data(old_x, observed, start=0.5):
    x = []
    new_observed = []
    # each entry in x is a list of all time prior than x
    for row in old_x.iterrows():
        index, data = row
        temp = list(data)
        time = data[0]
        step = start
        while step < time:
            new_row = temp[:]
            new_row[0] = step
            x.append(new_row)
            new_observed.append(0.0)
            step += start
        temp[0] = step
        x.append(temp)
        if observed[index]:
            new_observed.append(1.0)
        else:
            new_observed.append(0.0)
    return np.asarray(x), np.asarray(new_observed)


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
    # print np.asarray(test_series)[0]
    observed.replace(to_replace='A', value=0, inplace=True)
    observed.replace(to_replace='D', value=1, inplace=True)
    df.pop('vital.status')
    x, discrete_observed = discrete_time_data(df, observed)

    # df.insert(0, 'event_col', observed)
    # cf = CoxPHFitter()
    # cf.fit(df, 'Overall_Survival', 'event_col')
    # cf.survival_function_.plot(c='r')
    # print df.columns
    return x, discrete_observed, t_column, observed, np.asarray(test_series)[0]

if __name__ == '__main__':
    load_training_data('C:/Users/Song/Research/biomed/Survival/trainingData.csv')
