""" Classes to preprocess data for mercedes kaggle contest """
#!/usr/bin/env
import pandas as pd

def load_data(path, drop_cols=None):
    """ Helper function to load in data from csv"""
    if drop_cols:
        dataframe = pd.read_csv(path, header=0)
        dataframe = dataframe.drop(drop_cols, axis=1)
        return dataframe
    else:
        return pd.read_csv(path, header=0)

def get_cols(dfm):
    """ Helper function to get keep uniform columns across all sets """
    dataframe = pd.get_dummies(dfm)
    return list(dataframe.columns)

def group_list(l, group_size=batch_size):
    """ Generator to chunk data into batches """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]
