""" Script for preparing data  """
#!/usr/bin/env
import tensorflow as tf
import pandas as pd
import numpy as np
import helper_funcs as helpers
from mercedes_classes import DataFrameSelector, Dummifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Opening data
train_path = 'train.csv'

train = helpers.load_data(train_path, drop_cols=['ID'])
y_target = train["y"].values
train = train.drop(["y"], axis=1)

#Splitting the data into train and test sets
XX_train, XX_test, y_train, y_test = train_test_split(train, y_target, test_size=0.2, random_state=42)

#getting columns found in both data sets:
train_cols = helpers.get_cols(XX_train)
test_cols = helpers.get_cols(XX_test)
columns = [x for x in train_cols if x in test_cols]


#### Pipeline for processing data #######
pipeline = Pipeline([
        ('dummies', Dummifier()),
        ('selector', DataFrameSelector(columns)),
        ('std_scaler', StandardScaler()),
    ])  

X_train = pipeline.fit_transform(XX_train)
X_test = pipeline.transform(XX_test)
