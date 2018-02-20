#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:03:33 2018

@author: haikundu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#transform xlsx to dataframe
def xlsx_to_csv_pd(path):
    data_train = pd.read_excel(path + '15.xlsx')
    xls_16 = pd.read_excel(path + '16.xlsx')
    xls_17 = pd.read_excel(path + '17.xlsx')
    data_train = pd.concat([data_train,xls_16,xls_17])
    data_test = pd.read_excel(path + '18.xlsx')
    return data_train, data_test

def choose_column(df):
	count = df.shape[0] - df.count()
	filtered =  count[count < len(df)*0.2]
	index = list(filtered.index)
	index = [col for col in index if 'EPA' not in col]
	return df[index], index

def cate_encoding(df):
	#category = df.drop_duplicates([col]).toarray()
	#numbers = list(range(len(category)))
	cols = list(df.select_dtypes(include=['object']).columns)
	lb_make = LabelEncoder()
	for col in cols:
		try:
			df[col] = lb_make.fit_transform(df[col])
		except:
			print(col)
	return df

def fillna_mean(df,index):
    for col in index:
        if isinstance(df[col][0],(int,float)):
            df = df.fillna(df.mean()[col])
        else:
            df = df.fillna(df.mode()[col])
    return df

def fillna_median(df,index):
    for col in index:
        if isinstance(df[col][0],(int,float)):
            df = df.fillna(df.median()[col])
        else:
            df = df.fillna(df.mode()[col])
    return df


def fillna_knn(df):
	return 1

train, test = xlsx_to_csv_pd('/Users/haikundu/Desktop/4995AML/hw3/')
filtered_train, column_name = choose_column(train)
full_train = fillna_mean(filtered_train,column_name)
encoded_train = cate_encoding(full_train)

