#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:19:39 2018

@author: duxuewei
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

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


rf_full_train = rfimpute(filtered_train, full_train)