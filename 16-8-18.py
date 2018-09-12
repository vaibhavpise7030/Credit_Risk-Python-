# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:59:47 2018

@author: User
"""

#SVM
#%%
import pandas as pd
import numpy as np
#%%
#header=0
train_data=pd.read_csv(r'C:\Users\User\Desktop\R projects\CREDIT_RISK_TRAINING_DATA.csv')
test_data=pd.read_csv(r'C:\Users\User\Desktop\R projects\R_Module_Day_8.1_Credit_Risk_Test_data.csv')
#%%
print(train_data.shape)
train_data.head()
#%%
#find missing values
print(train_data.isnull().sum())
#print(train_data.shape)
#%%
#impute categorical missing data with mode value except credit history, index=0 4 mode
colname1=["Gender","Married","Dependents","Self_Employed","Loan_Amount_Term"]
for x in colname1[:]:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)
#%%
print(train_data.isnull().sum())
#%%
train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(),inplace=True)
print(train_data.isnull().sum())
#%%
#imputing values for credit history
train_data['Credit_History'].fillna(value=0,inplace=True)
print(train_data.isnull().sum())
#%%
#transfor categorical into numerical
from sklearn import preprocessing
colname=['Gender','Dependents','Married','Education','Self_Employed','Property_Area','Loan_Status']
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    train_data[x]=le[x].fit_transform(train_data.__getattr__(x))
#%%
print(train_data)
train_data.head()
#%%
print(test_data.isnull().sum())
print(test_data.shape)
#%%
colname1=["Gender","Dependents","Self_Employed","Loan_Amount_Term"]
for x in colname1[:]:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)
#%%
print(test_data.isnull().sum())
#%%
test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean(),inplace=True)
print(test_data.isnull().sum())
#%%
test_data['Credit_History'].fillna(value=0,inplace=True)
print(test_data.isnull().sum())
#%%
from sklearn import preprocessing
colname=['Gender','Dependents','Married','Education','Self_Employed','Property_Area']
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    test_data[x]=le[x].fit_transform(test_data.__getattr__(x))
#%%
print(test_data)
test_data.head()
#%%
#creating training and testing datasets
X_train=train_data.values[:,1:-1]
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)
#%%
#test_data.head()
X_test=test_data.values[:,1:]
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#fit only on train data
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#%%
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
#from sklearn.linear_model import LogisticRegression
#svc_model=LogisticRegression()
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))
#%%
Y_pred_col=list(Y_pred)
#%%
test_data=pd.read_csv(r'C:\Users\User\Desktop\R projects\R_Module_Day_8.1_Credit_Risk_Test_data.csv')
test_data["Y_predictions"]=Y_pred_col
test_data.head()
#%%
test_data.to_csv('creditrisktestdata.csv')
#%%
#to evaluate model we need confusion matrix, 4 confusion matrix we need y_test but v don't have that here
#so here v use cross validation
classifier=svm.SVC(kernel='rbf',C=1,gamma=0.1)
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%
#tune the model# linear u cannot change c and gamma
classifier=svm.SVC(kernel='linear',C=1,gamma=0.1)
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%
#try changing gamma and c values
classifier=svm.SVC(kernel='rbf',C=1,gamma=0.01)
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%
classifier=svm.SVC(kernel='poly',C=1,gamma=0.01)
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())



