import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for

data = pd.read_csv('/workspaces/ICT_Internship_Deploy/credit.csv')

data=data.drop('ID',axis=1)
data=data.drop('Customer_ID',axis=1)
data=data.drop('Name',axis=1)
data=data.drop('Age',axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cols = ['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour','Credit_Score']

for col in cols:
    data[col] = le.fit_transform(data[col])

X = data.drop(['Credit_Score'],axis=1)
y = data['Credit_Score']

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Assuming x and y are features and target variable
X = data[['Payment_Behaviour','Payment_of_Min_Amount','Credit_Mix','Occupation','Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan']]
y = data['Credit_Score']

ros = RandomOverSampler(random_state=42)

# Fit predictor and target variable
X_ros, y_ros = ros.fit_resample(X, y)

from sklearn.model_selection import train_test_split

X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X_ros, y_ros, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf_cls=RandomForestClassifier()
model_rf=rf_cls.fit(X_train_temp,y_train_temp)
y_pred_rf=model_rf.predict(X_val)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
acc = accuracy_score(y_val,y_pred_rf)
print(acc)

# Save the trained model using pickle
model_filename = 'credit_score.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(rf_cls, model_file)