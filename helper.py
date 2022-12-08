import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import Lab4_helper
import Lab5_helper

default = 0

def make_tree_rf(X,y,min_split_count=5, prevFeature = "", prevPrevFeature = "", type = "classifier"):
    tree = {}
    differentYVals = y.unique()
    defaultAns = y.value_counts().sort_values(ascending=False).index[0]

    #selecting a subset of the features
    numFeatures = int(np.round_(np.sqrt(X.shape[1])))
    X = X.sample(n = numFeatures, axis="columns")

    if type == "classifier":
        defaultAns = y.value_counts().sort_values(ascending=False).index[0]
    elif type == "regressor":
        defaultAns = y.mean()
    if(differentYVals.size) == 1:
        return defaultAns
    if(X.shape[0] < min_split_count):
        return defaultAns
    if(X.shape[1] == 0):
        return defaultAns
    if(X.drop_duplicates().shape[0] == 1):
        return defaultAns
    #then we're in the recursive case
    
    
    #pick the field with the highest IG
    bestFeature, rig = Lab4_helper.select_split2(X, y, type=type)
    if rig <= 0.001:
        return defaultAns
    
    #replacing the continous column with its split counterpart
    bestFeatureName = bestFeature.name
    originalColumnName = bestFeatureName.split("<")[0]
    X = X.drop(columns=[originalColumnName])
    X[bestFeatureName] = bestFeature

    #setting up tree
    tree[bestFeatureName] = {}
    possibleFeatureValues = [True, False]
    
    #recursing
    for value in possibleFeatureValues:
        XAtVal = X.loc[X[bestFeatureName] == value]
        YAtVal = y.loc[X[bestFeatureName] == value]     
        RestOfX = XAtVal.drop(columns=bestFeatureName)
        tree[bestFeatureName][str(value).capitalize()] = make_tree_rf(RestOfX, YAtVal, prevFeature=bestFeatureName, prevPrevFeature=prevFeature)
    
    return tree

def make_rf_trees(x, t, ntrees, type="classifier"):
    trees = []
    for _ in range(ntrees):
        #grabbing a random resample of the rows
        x_i, y_i = resample(x, t)
        tree = make_tree_rf(x_i, y_i, type=type)
        trees.append(tree)

    return trees

def make_rules(trees):
    rules = []
    for tree in trees:
        rule = Lab4_helper.generate_rules(tree)
        rules.append(rule)
    
    return rules

def make_pred_from_rules(x, rules, default = 0, type="classifier"):
    predictions = pd.DataFrame()
    for rule in rules:
        #making the prediction from a single truee
        pred = x.apply(lambda x: Lab4_helper.make_prediction(rule,x,default),axis=1)
        #adding it to our dataframe of predictions
        predictions = pd.concat([predictions, pred], axis = 1)

    #we want the prediction to be the most popular prediction from our subset of predictions for classification
    if type == "classifier":
        aggregate_prediction = predictions.mode(axis=1)
    #and for regression we want it to be the mean
    if type == "regressor":
        aggregate_prediction = predictions.mean(axis=1)
    return aggregate_prediction

def run_boost(X, t, results, type, ntrials, ntrees):
    scores = []
    for trial in range(ntrials):
        #splitting
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3,random_state=trial)
        #performing algorithm
        y = None
        model = None
        if type == "classifier":
            model = GradientBoostingClassifier(n_estimators=ntrees).fit(X_train, t_train)
        if type == "regressor":
            model = GradientBoostingClassifier(n_estimators=ntrees).fit(X_train, t_train)
        y = model.predict(X_test)

        #calculating performance
        if type == "classifier":
            scores.append(f1_score(y, t_test, average="weighted"))
        elif type == "regressor":
            scores.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
    average_score = sum(scores) / len(scores)
    results[f"boost_{type}"] = average_score

def run_skrf(X, t, results, type, ntrials, ntrees):
    scores = []
    for trial in range(ntrials):
        #splitting
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3,random_state=trial)
        #performing algorithm
        y = None
        if type == "classifier":
            model = RandomForestClassifier(n_estimators=ntrees, min_samples_split=5, random_state=trial, criterion="entropy").fit(X_train, t_train)
        if type == "regressor":
            model = RandomForestRegressor(n_estimators=ntrees, min_samples_split=5, random_state=trial, criterion="squared_error").fit(X_train, t_train)

        y = model.predict(X_test)
        #calculating performance
        if type == "classifier":
            scores.append(f1_score(y, t_test, average="weighted"))
        elif type == "regressor":
            scores.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
    average_score = sum(scores) / len(scores)
    results[f"skrf_{type}"] = average_score

def run_bagging(X, t, results, type, ntrials, ntrees):
    scores = []
    for trial in range(ntrials):
        #splitting
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3,random_state=trial)

        trees = Lab5_helper.make_trees(X_train, t_train,ntrees=ntrees, type=type)
        y = Lab5_helper.make_prediction(trees, X_test, type=type)

        #calculating performance
        if type == "classifier":
            scores.append(f1_score(y, t_test, average="weighted"))
        elif type == "regressor":
            scores.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
    average_score = sum(scores) / len(scores)
    results[f"bagging_{type}"] = average_score

def run_myrf(X, t, results, type, ntrials, ntrees):
    scores = []
    for trial in range(ntrials):
        #splitting
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3,random_state=trial)
        trees = make_rf_trees(X_train, t_train, ntrees=ntrees, type=type)
        rules = make_rules(trees)
        y = make_pred_from_rules(X_test, rules, default=default, type=type)

        #calculating performance
        if type == "classifier":
            scores.append(f1_score(y, t_test, average="weighted"))
        elif type == "regressor":
            scores.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
    average_score = sum(scores) / len(scores)
    results[f"myrf_{type}"] = average_score