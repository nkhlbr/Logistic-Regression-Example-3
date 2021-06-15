# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:20:21 2021

@author: nikhil.barua
"""


#Heavily focused on graph

#Exploring Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        """


df = pd.read_csv('HR_comma_sep.csv')
df.head()



#Data Inspection

df.describe()


print('Existence of null values: ', df.isnull().values.any())
print('Existence of NaN values: ', df.isna().values.any())

df['left'].value_counts(normalize=True)



department_list=df['Department'].value_counts()
ret_ratio=df.groupby('Department')['left'].value_counts()
ratio_arr=np.zeros(len(department_list))

 

department_list
ret_ratio
ratio_arr


i=0
for j in department_list.keys():
    #print(j,'--> Stay: ',ret_ratio[j][0],'Left: ',ret_ratio[j][1])
    ratio_arr[i]=100*ret_ratio[j][1]/(ret_ratio[j][0]+ret_ratio[j][1])
    i=i+1

salary_list=df['salary'].value_counts()

sal_ratio=df.groupby('salary')['left'].value_counts()
sal_arr=np.zeros(len(salary_list))

i=0
for j in salary_list.keys():
    #print(j,'--> Stay: ',ret_ratio[j][0],'Left: ',ret_ratio[j][1])
    sal_arr[i]=100*sal_ratio[j][1]/(sal_ratio[j][0]+sal_ratio[j][1])
    i=i+1


fig,ax = plt.subplots(ncols=2,figsize=(20,5))

plt.sca(ax[0])


_rt_bar=sns.barplot(x=department_list.keys(), y=ratio_arr)
_rt_title=plt.title('Resignation Rate Per Department')

for bar in _rt_bar.patches:
    _rt_bar.annotate(format(bar.get_height(), '.2f'),  
               (bar.get_x() + bar.get_width() / 2,  
                bar.get_height()), ha='center', va='center', 
               size=12, xytext=(0, 8), 
               textcoords='offset points') 

_rt_xtick=plt.xticks(rotation=45)
_rt_ylim=plt.ylim(0,100)
_rt_ylabel=plt.ylabel('%')
               

fig,ax = plt.subplots(ncols=3,figsize=(20,5))
_box=sns.boxplot(data = df,y='satisfaction_level',x='left',showmeans=True,ax=ax[0])
_box=sns.boxplot(data = df,y='last_evaluation',x='left',showmeans=True,ax=ax[1])
_box=sns.boxplot(data = df,y='average_montly_hours',x='left',showmeans=True,ax=ax[2])
for n in range(0,3):
    ax[n].set_xticklabels(labels=['Stayed','Left'])
    ax[n].set_xlabel(None)

fig,ax = plt.subplots(ncols=2,figsize=(10,5))
_box=sns.boxplot(data = df,y='time_spend_company',x='left',showmeans=True,ax=ax[0])
_box=sns.boxplot(data = df,y='number_project',x='left',showmeans=True,ax=ax[1])
for n in range(0,2):
    ax[n].set_xticklabels(labels=['Stayed','Left'])
    ax[n].set_xlabel(None)


sns.pairplot(df, hue="left")

fig,ax=plt.subplots(ncols=3,figsize=(20,5))
sns.scatterplot(data=df,x='satisfaction_level',y='last_evaluation',hue='left', ax=ax[0])
sns.scatterplot(data=df,x='satisfaction_level',y='average_montly_hours',hue='left', ax=ax[1])
sns.scatterplot(data=df,x='last_evaluation',y='average_montly_hours',hue='left', ax=ax[2])



fig=plt.figure(figsize=(10,5))
sns.scatterplot(data=df,x='satisfaction_level',y='last_evaluation',hue='left')
plt.vlines(0.675,0.75,1.0,'red')
plt.hlines(0.75,0.675,0.95,'red')
plt.vlines(0.95,0.75,1.0,'red')


df_x=df.loc[(df["left"] == 1) & (df["last_evaluation"] > 0.7) & (df["satisfaction_level"]>0.6)]
print('Resigned Cluster Average - Overall Average')
print(df_x.mean()-df.mean())
print('-----------------------------')
print('Resigned Cluster Salary Range')
print(df_x['salary'].value_counts())




#Logistic Regression

fig,ax=plt.subplots(ncols=2,figsize=(20,8))
resign_corr=df.corr()
mask = np.triu(np.ones_like(resign_corr, dtype=np.bool))
cat_heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',ax=ax[0])
cat_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

heatmap = sns.heatmap(resign_corr[['left']].sort_values(by='left', ascending=False),vmin=-1, vmax=1, annot=True, cmap='BrBG',ax=ax[1])
heatmap.set_title('Features Correlating with Resignation', fontdict={'fontsize':18}, pad=16);


#One Hot Encoding what is it

df_lr=df.copy()
df_lr=pd.get_dummies(df_lr, columns = ['Department','salary'])
df_lr.head()

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression


X = np.asarray(df_lr.loc[:, df_lr.columns != 'left'])
y = np.asarray(df_lr.loc[:, df_lr.columns == 'left'])

X
y


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.metrics import roc_auc_score,roc_curve
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train.ravel())

y_pred=model.predict(X_test)
y_proba=model.predict_proba(X_test)

ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
print("ROC AUC SCORE: ",roc_auc_score(y_test, y_proba[:, 1]))
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_proba[:,1])
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


cf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(cf_matrix, annot=True, cmap='Blues')

print(classification_report(y_test, y_pred))


from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gb_clf.fit(X_train, y_train.ravel())

y_gb=gb_clf.predict(X_test)

gb_matrix=confusion_matrix(y_test, y_gb)
sns.heatmap(gb_matrix, annot=True, cmap='Blues')
print(classification_report(y_test, y_gb))




































