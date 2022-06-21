import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import yaml
import pickle

#create folder to save file 
data_path = os.path.join ('data','prepared')
os.makedirs(data_path, exist_ok=True)

#load data
df = pd.read_csv('src/client1.csv')
#preprocessing
df = df.drop(['fnlwgt', 'educational-num'], axis=1)
col_names = df.columns
for c in col_names:
    df = df.replace("?", np.nan)
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
#structrure the categorical values
df.replace(['Divorced', 'Married-AF-spouse',
            'Married-civ-spouse', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)

category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']



 #encode label
labelEncoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
 
    le_name_mapping = dict(zip(labelEncoder.classes_,
                               labelEncoder.transform(labelEncoder.classes_)))
 
    mapping_dict[col] = le_name_mapping

df = df.drop(['race'],axis=1)
#save the prepared data in the specific path in csv file
df.to_csv(os.path.join(data_path,'newdata.csv'))
