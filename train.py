#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from joblib import dump

#import dataset
diabets = pd.read_csv("datasets/diabetes.csv")
diabets = diabets.drop("DiabetesPedigreeFunction", axis=1)
diabets = diabets.drop("SkinThickness", axis=1)
diabets_cp = diabets

#split to train and test sets
train_set, test_set = train_test_split(diabets, test_size=0.2, random_state=42)
train_labels = train_set['Glucose'].copy()
train_set = train_set.drop("Glucose",axis=1)
test_labels = test_set['Glucose'].copy()
test_set = test_set.drop("Glucose",axis=1)

#Fix scales
p = Pipeline([('std',StandardScaler())])
train_set = p.fit_transform(train_set)
test_set = p.transform(test_set)

#RandomForest
forest_reg = RandomForestRegressor()
#train model
forest_reg = forest_reg.fit(train_set, train_labels)

#Fine Tune model using gridsearchcv
param_grid = [{'n_estimators':[100,200,300],'max_features':[2,4,6]},{'bootstrap':[False],'n_estimators':[100,200],'max_features':[2,4,6]}]
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
#train gridsearchcv
grid_search = grid_search.fit(train_set,train_labels)
grid_search = grid_search.best_estimator_



