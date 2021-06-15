# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:31:16 2021

@author: nikhil.barua
"""

#lOGISTIC REGRESSION ON WHY HR IS LEAVING

#Data Exploration


import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('HR_comma_sep.csv')

df.info()
df.head()
df.tail()

df.shape #returns a tuple representing the dimensionality

df.ndim #returns dimensions of array dataset

df.size #no.of elements

df.columns #list the column names

df.describe() #gives descriptive statistics

df.isnull().sum() #check for null values in dataset

df.duplicated().sum()

df.drop_duplicates(inplace=True) #drop duplicate rows

df.duplicated().sum()

df.info()

df.shape

df.nunique()

#Data visuailization

relation = df.corr()
relation


plt.figure(figsize=(10,6))
sns.heatmap(relation, annot=True)

#Data Preprocessing

new_df = df[['satisfaction_level', 'average_montly_hours', 'time_spend_company','promotion_last_5years', 'salary' ]] 

new_df

salary = pd.get_dummies(new_df['salary'], prefix='salary')  #what does get dummies do--convert string into the values for the model to predict
salary

df_dummies = pd.concat([new_df, salary],axis=1)

df_dummies.head()

df_dummies.drop('salary', axis=1,inplace=True)  #drop the salary string column
df_dummies.head()

#Preparing the data

x = df_dummies
x

y = df['left']
y



#Training the dataset 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state = 1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Feature Scaling   inorder to increase the accurate rate ofpredictions

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train = st_x.fit_transform(X_train)
x_test = st_x.transform(X_test)


#Apply Logistic Regression model

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)


predictions = log_model.predict(x_test)
predictions


#comparing actual and predicted values

data = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
data.head(20)


#Testing the accuracy

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
cm=confusion_matrix(y_test, predictions)
cm


log_model.score(x_test,y_test)

#Evaluation metrics

print('\n')
print('Accuracy  : %0.4f ' % accuracy_score(y_test,predictions))
print('Precision  : %0.4f ' % precision_score(y_test,predictions))
print('Recall  : %0.4f ' % recall_score(y_test,predictions))

#Classification report



print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))

















