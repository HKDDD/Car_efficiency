#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:03:33 2018

@author: haikundu
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import seaborn as sns

#transform xlsx to dataframe
def xlsx_to_csv_pd(path):
    data_train = pd.read_excel(path + '15.xlsx')
    xls_16 = pd.read_excel(path + '16.xlsx')
    xls_17 = pd.read_excel(path + '17.xlsx')
    data_train = pd.concat([data_train,xls_16,xls_17])
    data_test = pd.read_excel(path + '18.xlsx')
    return data_train, data_test

def x_y_split(df):
    y = df.pop('Comb Unrd Adj FE - Conventional Fuel')
    return df,y

def choose_column(df):
	count = df.shape[0] - df.count()
	filtered =  count[count < len(df)*0.2]
	index = list(filtered.index)
	index = [col for col in index if col.find('EPA') == -1 and col.find('FE') == -1 and col.find('MPG') == -1 and col.find('CO2') == -1 and col.find('Smog') == -1 and col.find('Guzzler') == -1 and col.find('Release Date') == -1 and col.find('Mfr Name') == -1 and col.find('Verify Mfr Cd') == -1 and col.find('Fuel Unit - Conventional Fuel') == -1]
	index = [col for col in index if col.find('Desc') == -1 or col.find('Calc Approach Desc') > -1 or col.find('Var Valve Timing Desc') > -1]
	return df[index], index

def fillna_mean(dfo,index,dfto):
    df = dfo.copy()
    dft = dfto.copy()
    for col in index:
        if isinstance(df[col][0],(int,float)):
            dfo.loc[:,col] = df[col].fillna(df.mean()[col].tolist()[0]).tolist()
            dfto.loc[:,col] = dft[col].fillna(df.mean()[col].tolist()[0]).tolist()
        else:
            dfo.loc[:,col] = df[col].fillna(df.mode()[col].tolist()[0]).tolist()
            dfto.loc[:,col] = dft[col].fillna(df.mode()[col].tolist()[0]).tolist()
    return dfo,dfto

def fillna_median(dfo,index,dfto):
    df = dfo.copy()
    dft = dfto.copy()
    for col in index:
        if isinstance(df[col][0],(int,float)):
            dfo.loc[:,col] = df[col].fillna(df.median()[col].tolist()[0]).tolist()
            dfto.loc[:,col] = dft[col].fillna(df.median()[col].tolist()[0]).tolist()
        else:
            dfo.loc[:,col] = df[col].fillna(df.mode()[col].tolist()[0]).tolist()
            dfto.loc[:,col] = dft[col].fillna(df.mode()[col].tolist()[0]).tolist()
    return dfo,dfto

def one_hot_encoding(train,test):
    full = pd.concat([train, test])
    extended = pd.get_dummies(full)
    boundry = len(train)
    train = extended[:boundry]
    test = extended[boundry:]
    return train, test

train, test = xlsx_to_csv_pd('/Users/haikundu/Desktop/4995AML/hw3/')
train_x, train_y = x_y_split(train)
test_x, test_y = x_y_split(test)
filtered_train, column_name = choose_column(train_x)
filtered_test = test[column_name]
full_train,full_test = fillna_mean(filtered_train,column_name,filtered_test)

#one hot encoding
ohe_train,ohe_test = one_hot_encoding(full_train, full_test)

#Standard Scaler
scaler = StandardScaler()
scaler.fit(ohe_train)
X_train_scaled = scaler.transform(ohe_train)
X_test_scaled = scaler.transform(ohe_test)

#ridge regression
ridge = Ridge().fit(X_train_scaled, train_y)
ridge.score(X_test_scaled, test_y)

#Lasso regression


#ElasticNet

#make pipeline
#all_features = make_pipeline(StandardScaler(), RidgeCV())
print(np.mean(cross_val_score(RidgeCV(), X_train_scaled, train_y, cv= 10)))
