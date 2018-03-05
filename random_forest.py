#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:59:48 2018

@author: duxuewei
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


#random forest
#train_scores = []
#test_scores = []
#rf = RandomForestRegressor(warm_start=True)
#estimator_range = range(1, 100, 5)
#for n_estimators in estimator_range:
#    rf.n_estimators = n_estimators
#    rf.fit(X_train_scaled, train_y)
#    train_scores.append(rf.score(X_train_scaled, train_y))
#    test_scores.append(rf.score(X_test_scaled, test_y))
    

#scores = cross_val_score(RandomForestRegressor(), X_train_scaled, train_y, cv= 10)
#np.mean(scores)

rf_reg = RandomForestRegressor()

param_grid = {
                 'n_estimators': [5, 10, 15, 20],
                 'max_depth': [2, 5, 7, 9, 11, 13]
             }

grid_rf_reg = GridSearchCV(rf_reg, param_grid, cv=10)
grid_rf_reg.fit(X_train_scaled, train_y)


#grid_rf_reg.best_estimator_

print("Best parameters for random forest regression model are "+str(grid_rf_reg.best_params_))

print("Best random forest training score using gridsearch is "+str(grid_rf_reg.best_score_))

#grid_rf_reg.cv_results_


#rf_best = grid_rf_reg.best_estimator_

#rf_score = rf_best.score(X_test_scaled, test_y)
#print("Best random forest test score is "+ str(rf_score))

#rf_feature_importances = rf_best.feature_importances_
