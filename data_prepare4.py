#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model

raw_train = pd.read_csv('input/train.csv')
raw_train['is_test'] = 0
raw_test = pd.read_csv('input/test.csv')
raw_test['is_test'] = 1

all_data = pd.concat((raw_train, raw_test), axis=0, sort=True)

# Functions to preprocess the data

def get_title_last_name(name):
    full_name = name.str.split(', ', n=0, expand=True)
    last_name = full_name[0]
    titles = full_name[1].str.split('.', n=0, expand=True)
    titles = titles[0]
    return(titles)

def get_titles_from_names(df):
    df['Title'] = get_title_last_name(df['Name'])
    df = df.drop(['Name'], axis=1)
    return(df)

def get_dummy_cats(df):
    return(pd.get_dummies(df, columns=['Title', 'Pclass', 'Sex', 'Embarked',
                                       'Cabin', 'Cabin_letter']))

def get_cabin_letter(df):    
    df['Cabin'].fillna('Z', inplace=True)
    df['Cabin_letter'] = df['Cabin'].str[0]    
    return(df)

def process_data(df):
    # preprocess titles, cabin, embarked
    df = get_titles_from_names(df)    
    df['Embarked'].fillna('S', inplace=True)
    df = get_cabin_letter(df)
    
    # drop remaining features
    df = df.drop(['Ticket', 'Fare'], axis=1)
    
    # create dummies for categorial features
    df = get_dummy_cats(df)
    
    return(df)

proc_data = process_data(all_data)
proc_train = proc_data[proc_data['is_test'] == 0]
proc_test = proc_data[proc_data['is_test'] == 1]

proc_data.head()

# Build Network to predict missing ages

for_age_train = proc_data.drop(['PassengerId', 'Survived', 'is_test'], axis=1).dropna(axis=0)
X_train_age = for_age_train.drop('Age', axis=1)
y_train_age = for_age_train['Age']

# create model
'''
tmodel = Sequential()
tmodel.add(Dense(input_dim=X_train_age.shape[1], units=128,
                 kernel_initializer='normal', bias_initializer='zeros'))
tmodel.add(Activation('relu'))

for i in range(0, 8):
    tmodel.add(Dense(units=64, kernel_initializer='normal',
                     bias_initializer='zeros'))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(.25))

tmodel.add(Dense(units=1))
tmodel.add(Activation('linear'))

tmodel.compile(loss='mean_squared_error', optimizer='rmsprop')

# train model
tmodel.fit(X_train_age.values, y_train_age.values, epochs=600, verbose=2)

tmodel.save('age.h5')

sys.exit(0)
'''

tmodel = load_model('age.h5')

train_data = proc_train
train_data.loc[train_data['Age'].isnull()]

# predict age
to_pred = train_data.loc[train_data['Age'].isnull()].drop(
        ['PassengerId', 'Age', 'Survived', 'is_test'], axis=1)
p = tmodel.predict(to_pred.values)

train_data['Age'].loc[train_data['Age'].isnull()].shape
train_data['Age'].loc[train_data['Age'].isnull()] = p.reshape(177,)
#train_data['Age'].loc[train_data['Age'].isnull()] = p


test_data = proc_test
to_pred = test_data.loc[test_data['Age'].isnull()].drop(
        ['PassengerId', 'Age', 'Survived', 'is_test'], axis=1)
p = tmodel.predict(to_pred.values)

test_data['Age'].loc[test_data['Age'].isnull()].shape
test_data['Age'].loc[test_data['Age'].isnull()] = p.reshape(86,)

# 年龄都填充
train_data.loc[train_data['Age'].isnull()]
test_data.loc[test_data['Age'].isnull()]


# 数据准备
#test_submission = pd.read_csv('./input/gender_submission.csv')
test_submission = pd.read_csv('input/target.csv')
test_submission.head()

#x_train0 = train_data.drop(['Survived', 'is_test'], axis=1)
#y_train0 = train_data['Survived']
#
#x_test0 = test_data.drop(['Survived', 'is_test'], axis=1)
#y_test0 = test_submission['Survived']
#
#x_train = x_train0.values.astype('float32')
#x_test = x_test0.values.astype('float32')
#
#y_train = y_train0.values.astype('float32')
#y_test = y_test0.values.astype('float32')


x_train = train_data.drop(['PassengerId', 'Survived', 'is_test'], axis=1)
y_train = pd.get_dummies(train_data['Survived'])

x_test = test_data.drop(['PassengerId', 'Survived', 'is_test'], axis=1)
y_test = pd.get_dummies(test_submission['Survived'])


test = pd.read_csv('./input/test.csv')