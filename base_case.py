import pandas as pd
import numpy as np
import collections
import xgboost
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix    
from collections import Counter

data=pd.read_csv("sensor.csv")

data= data.drop(['timestamp','sensor_15'],axis=1)

#fill NAN by mean value
data=data.fillna(data.mean())

target = data.loc[:, data.columns == 'machine_status']
print("original  data size",collections.Counter(target['machine_status']))
num_target=collections.Counter(target['machine_status'])
print("Noraml",num_target['NORMAL']/len(target))
print("Recovering",num_target['RECOVERING']/len(target))
print("Broken",num_target['BROKEN']/len(target))

X = data.loc[:, data.columns != 'machine_status']
y = data.loc[:, data.columns == 'machine_status']

#split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,train_size=0.7, random_state = 0)

#xgb
model = xgboost.XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Predicted  ",collections.Counter(y_pred))
print("testing data size",collections.Counter(y_test['machine_status']))
cn_mat=confusion_matrix( y_pred,y_test['machine_status'])
print(cn_mat)
