import copy
import json

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def get_learner(X, y, type, max_depth=10):
    if type == "regressor":
        return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)
    if type == "classifier":
        return DecisionTreeClassifier(max_depth=max_depth).fit(X,y)
    print("unsupported type given", type)

def make_trees(X,y,ntrees=100,max_depth=10, type="regressor", random_forest=False):
    trees = []
    for i in range(ntrees):
        # Your solution here
        x_i, y_i = resample(X, y)
        
        #taking a subsample of the features if we're doing a random forest
        if random_forest:
            numFeatures = int(np.round_(x_i.shape[1] / 3))
            x_i = x_i.sample(n = numFeatures, axis="columns")

        tree = get_learner(x_i, y_i, type=type)
        trees.append(tree)
    return trees

def make_prediction(trees,X, type="regressor"):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        #only when we use random trees will the features the classifier has learned will be not be all the features
        tree_predictions.append(tree.predict(X[tree.feature_names_in_]).tolist())
    if type == "regressor":
        return np.array(pd.DataFrame(tree_predictions).mean().values.flat)
    elif type == "classifier":
        return np.array(pd.DataFrame(tree_predictions).mode().iloc[0].values.flat)

def make_trees_boost(Xtrain, Xval, ytrain, yval, max_ntrees=100,max_depth=2, type="regressor"):
    trees = []
    yval_pred = None
    ytrain_pred = None
    train_RMSEs = [] # the root mean square errors for the validation dataset
    val_RMSEs = [] # the root mean square errors for the validation dataset
    ytrain_orig = copy.deepcopy(ytrain)
    for i in range(max_ntrees):
        #making learner and storing it
        tree = get_learner(Xtrain, ytrain, max_depth=max_depth, type=type)
        trees.append(tree)
        
        #making predictions with new learners
        ytrain_pred = tree.predict(Xtrain)
        yval_pred = tree.predict(Xval)
        
        #seeing how far off we are
        ytrain = ytrain - ytrain_pred
        yval = yval - yval_pred

        #storing RMSEs        
        train_RMSE = np.sqrt(((ytrain)**2).sum()/len(Xtrain))
        train_RMSEs.append(train_RMSE)
        val_RMSE = np.sqrt(((yval)**2).sum()/len(Xval))
        val_RMSEs.append(val_RMSE)
    
    return trees,train_RMSEs,val_RMSEs

def cut_trees(trees,val_RMSEs):

    # Your solution here that finds the minimum validation score and uses only the trees up to that
    minRMSEIndex = np.argmin(val_RMSEs) 
    return trees[:minRMSEIndex + 1]

def make_prediction_boost(trees,X, type="regressor"):
    tree_predictions = []
    for tree in trees:
        tree_predictions.append(tree.predict(X).tolist())
    
    return np.array(pd.DataFrame(tree_predictions).sum().values.flat)

