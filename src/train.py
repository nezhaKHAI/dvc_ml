import sys
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


data_path = os.path.join ('data','test')
os.makedirs(data_path, exist_ok=True)
#read parameters
params = yaml.safe_load(open('params.yaml'))['train']

test_size = params['test_size']
random_state = params['random_state']
criterion = params['criterion']
max_depth = params['max_depth']
min_samples_leaf = params['min_samples_leaf']
#read the data 
df_train = pd.read_csv('data/prepared/newdata.csv')

X = df_train.values[:, 0:12]
Y = df_train.values[:, 12]
#split and train the model
X_train, X_test, y_train, y_test = train_test_split(
           X, Y, test_size = test_size, random_state = random_state)
print(y_test)
 
dt_clf_gini = DecisionTreeClassifier(criterion = criterion,
                                     random_state = random_state,
                                     max_depth = max_depth,
                                     min_samples_leaf = min_samples_leaf)
 
dt_clf_gini.fit(X_train, y_train)
pickle.dump(X_test, open(os.path.join(data_path,'x_test.pkl'), 'wb'))
pickle.dump(y_test, open(os.path.join(data_path,'y_test.pkl'), 'wb'))

pickle.dump(dt_clf_gini, open('model_dt.pkl', 'wb'))





