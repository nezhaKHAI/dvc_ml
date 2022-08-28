
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, spearmanr
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
import os
import pickle
import sys 

data_path = os.path.join ('data','features')
os.makedirs(data_path, exist_ok=True)

data = pd.read_csv('data/prepared/newdata.csv')
col_names = data.columns

param=[]
correlation=[]
abs_corr=[]

for c in col_names:
    #Check if binary or continuous
    if c != "income":
        if len(data[c].unique()) <= 2:
            corr = spearmanr(data['income'],data[c])[0]
        else:
            corr = pointbiserialr(data['income'],data[c])[0]
        param.append(c)
        correlation.append(corr)
        abs_corr.append(abs(corr))
#Create dataframe for visualization
param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})

#Sort by absolute correlation
param_df=param_df.sort_values(by=['abs_corr'], ascending=False)

#Set parameter name as index
param_df=param_df.set_index('parameter')

param_df

scoresCV = []
scores = []

for i in range(1,len(param_df)):
    new_df=data[param_df.index[0:i+1].values]
    X = new_df.values[:,1::]
    y = new_df.values[:,0]
    clf = DecisionTreeClassifier()
    scoreCV = cross_val_score(clf, X, y, cv= 10)
    scores.append(np.mean(scoreCV))
best_features=param_df.index[0:4].values
features=list(best_features)
print("Best_features:",features)
pickle.dump(features, open(os.path.join(data_path,'best_features.pkl'), 'wb'))


