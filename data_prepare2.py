#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import pandas as pd    
import numpy as np
# Reading data to Dataframes
    
data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')
data_check = pd.read_csv('input/gender_submission.csv')

#changing column name
data_train = data_train.rename(columns = {'Pclass' : 'TicketClass'})
data_test = data_test.rename(columns = {'Pclass' : 'TicketClass'})

#Removing unused columns
data_train = data_train.drop(['Name','Ticket','Fare','Cabin','Embarked','Age'],axis =1)
data_test = data_test.drop(['Name','Age','Ticket','Fare','Cabin','Embarked'], axis =1)

#Importing LabelEncoder from Sklearn
from sklearn.preprocessing import LabelEncoder
label_encoder_sex = LabelEncoder()

# Transforming sex column values using label Encoder
data_train.iloc[:,3]  = label_encoder_sex.fit_transform(data_train.iloc[:,3])
data_test.iloc[:,2] = label_encoder_sex.fit_transform(data_test.iloc[:,2])

data_train = data_train[['PassengerId','Sex','SibSp','Parch','TicketClass','Survived']]
data_test = data_test[['PassengerId','Sex','SibSp','Parch','TicketClass']]

x_train = data_train.iloc[:,0:5]   # Inputs
y_train = data_train.iloc[:,5]     # Output (Survived)
y_train = y_train.values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(data_test)

y_test = data_check['Survived']
y_test = y_test.values

test = data_test
