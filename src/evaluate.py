from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import sys
from sklearn.metrics import accuracy_score
import json


#load data for test

x_test = pickle.load( open( "data/test/x_test.pkl", "rb") )
y_test = pickle.load( open( "data/test/y_test.pkl", "rb") )

#load  the model 
model = pickle.load( open( "model_dt.pkl", "rb") ) 

#make predictions        
y_pred_gini = model.predict(x_test)
print(y_test)

#calculate the accuracy
accuracy=accuracy_score(y_test, y_pred_gini)*100 

# save accuracy
with open('accuracy_file.json', 'w') as f:
    json.dump({'acc': accuracy}, f)

# save plot
with open('plots_file.json', 'w') as f:
    proc_dict = {'proc': [{
        'accuracy': accuracy
        
        } 
    ]}
    json.dump(proc_dict, f)

