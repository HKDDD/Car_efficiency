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
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

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
	index = [col for col in index if col.find('EPA') == -1 and col.find('FE') == -1 and col.find('MPG') == -1 and col.find('CO2') == -1 and col.find('Smog') == -1 and col.find('Guzzler') == -1 and col.find('Release Date') == -1 and col.find('Mfr Name') == -1 and col.find('Verify Mfr Cd') == -1]
	index = [col for col in index if col.find('Desc') == -1 or col.find('Calc Approach Desc') > -1 or col.find('Var Valve Timing Desc') > -1]
	return df[index], index

def fillna_mean(dfo,index,dfto):
    df = dfo.copy()
    dft = dfto.copy()
    for col in index:
        if isinstance(df[col][0],(int,float)):
            df.loc[:,col] = df[col].fillna(df.mean()[col].tolist()[0]).tolist()
            dft.loc[:,col] = dft[col].fillna(df.mean()[col].tolist()[0]).tolist()
        else:
            df.loc[:,col] = df[col].fillna(df.mode()[col].tolist()[0]).tolist()
            dft.loc[:,col] = dft[col].fillna(df.mode()[col].tolist()[0]).tolist()
    return df,dft

def fillna_median(dfo,index,dfto):
    df = dfo.copy()
    dft = dfto.copy()
    for col in index:
        if isinstance(df[col][0],(int,float)):
            df.loc[:,col] = df[col].fillna(df.median()[col].tolist()[0]).tolist()
            dft.loc[:,col] = dft[col].fillna(df.median()[col].tolist()[0]).tolist()
        else:
            df.loc[:,col] = df[col].fillna(df.mode()[col].tolist()[0]).tolist()
            dft.loc[:,col] = dft[col].fillna(df.mode()[col].tolist()[0]).tolist()
    return df,dft

def one_hot_encoding(train,test):
    full = pd.concat([train, test]) 
    extended = pd.get_dummies(full)
    boundry = len(train)
    train = extended[:boundry]
    test = extended[boundry:]
    return train, test

def rfimpute(df, full_df):
    """
    rfimpute imputes missing values in df using random forest regression.
    
    Input:
        df: a pandas data frame df with missing value, note that choosing
            columns must happen before imputing. 
        full_df: a pandas data frame with missing values filled using mean, 
                median or mode.
    Output: 
        a pandas data frame having the same dimension as df
    """
    rf_reg = RandomForestRegressor(n_estimators=100)
    rf_cla = RandomForestClassifier(n_estimators=100)
    X_imputed = full_df.copy()
    filtered_train = df.copy()
    
    for i in range(10):
        X_imputed_dummies = pd.get_dummies(X_imputed)
        last = X_imputed_dummies.copy().as_matrix()
        for feature in range(filtered_train.shape[1]):
            feature_name = filtered_train.columns.values[feature]
            
            #inds_not_f are column numbers that do not belong to feature column in dummies df
            inds_not_f = np.array([col for col in range(X_imputed_dummies.shape[1])\
              if (feature_name + "_") not in X_imputed_dummies.columns.values[col]])
            if len(inds_not_f) == X_imputed_dummies.shape[1]: #means feature is numerical
                  inds_not_f = np.array([col for col in range(X_imputed_dummies.shape[1])\
              if (feature_name) not in X_imputed_dummies.columns.values[col]])
        
            f_missing = filtered_train.isnull()[feature_name].values
            if any(f_missing): 
            
                #Convert df's to numpy matrix
                filtered_train_colnames = filtered_train.columns.values
                filtered_train_dtypes_dict = filtered_train.dtypes.to_dict()
                filtered_train = filtered_train.as_matrix()
                X_imputed_colnames = X_imputed.columns.values
                X_imputed_dtypes_dict = X_imputed.dtypes.to_dict()
                X_imputed = X_imputed.as_matrix()
                X_imputed_dummies_colnames = X_imputed_dummies.columns.values
                X_imputed_dummies = X_imputed_dummies.as_matrix()
                
                if len(inds_not_f) == X_imputed_dummies.shape[1] - 1: #numerical columns do not expand
                    rf_reg.fit(X_imputed_dummies[~f_missing][:, inds_not_f], filtered_train[~f_missing, feature])
                    X_imputed[f_missing, feature] = rf_reg.predict(
                            X_imputed_dummies[f_missing][:, inds_not_f])
                else: #for categorical a feature column
                    LabEnc = LabelEncoder()
                    y = LabEnc.fit_transform(filtered_train[~f_missing, feature])
                    rf_cla.fit(X_imputed_dummies[~f_missing][:, inds_not_f], y)
                    rf_cla_predicted_Enc = rf_cla.predict(X_imputed_dummies[f_missing][:, inds_not_f])
                    rf_cla_predicted = LabEnc.inverse_transform(rf_cla_predicted_Enc)
                    X_imputed[f_missing, feature] = rf_cla_predicted
                #Convert numpy matrix back to df's
                filtered_train = pd.DataFrame(data = filtered_train, columns = filtered_train_colnames)
                filtered_train = filtered_train.astype(filtered_train_dtypes_dict)
                X_imputed = pd.DataFrame(data = X_imputed, columns = X_imputed_colnames)
                X_imputed = X_imputed.astype(X_imputed_dtypes_dict)
                X_imputed_dummies = pd.get_dummies(X_imputed)

            
        now = pd.get_dummies(X_imputed).as_matrix()
        if (np.linalg.norm(last - now)) < .5:
            return X_imputed
    return X_imputed


train, test = xlsx_to_csv_pd('/Users/haikundu/Desktop/4995AML/hw3/')
train_x, train_y = x_y_split(train)
test_x, test_y = x_y_split(test)
filtered_train, column_name = choose_column(train_x)
filtered_test = test[column_name]
full_train,full_test = fillna_mean(filtered_train,column_name,filtered_test)

#fillna with random forest regression
full_train = rfimpute(filtered_train, full_train)
full_test = rfimpute(filtered_test, full_test)


#one hot encoding
ohe_train,ohe_test = one_hot_encoding(full_train, full_test)

#Standard Scaler
scaler = StandardScaler()
scaler.fit(ohe_train)
X_train_scaled = scaler.transform(ohe_train)
X_test_scaled = scaler.transform(ohe_test)

#ridge regression
ridge = Ridge().fit(X_train_scaled, train_y)
print(ridge.score(X_test_scaled, test_y))

#Lasso regression


#ElasticNet

#make pipeline
#all_features = make_pipeline(StandardScaler(), RidgeCV())
print(np.mean(cross_val_score(RidgeCV(), X_train_scaled, train_y, cv= 10)))
