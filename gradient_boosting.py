#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:13:12 2018

@author: duxuewei
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

#gradient boosting
gbm_reg = GradientBoostingRegressor()

param_grid = {
                 'n_estimators':range(50,71,10), 
                 'max_depth':range(18,25,3), 
                 'min_samples_split':range(250,950,300), 
                 'max_features':range(24,33,4) 
             }
#param_grid = {
#                 'n_estimators':range(40,71,10), 
#                 'max_depth':range(15,25,3), 
#                 'min_samples_split':range(250,1250,300), 
#                 'max_features':range(20,33,4) 
#             }

grid_gbm_reg = GridSearchCV(gbm_reg, param_grid, cv=10)
grid_gbm_reg.fit(X_train_scaled, train_y)


grid_gbm_reg.best_estimator_

grid_gbm_reg.best_params_

print("Best gradient boosting training score using gridsearch is "+str(grid_gbm_reg.best_score_))

grid_gbm_reg.cv_results_


gbm_best = grid_gbm_reg.best_estimator_

gbm_score = gbm_best.score(X_test_scaled, test_y)
print("Best gradient boosting test score is "+ str(gbm_score))

gbm_feature_importances = gbm_best.feature_importances_