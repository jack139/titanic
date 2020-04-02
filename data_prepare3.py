#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import os
print(os.listdir("./input"))

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
#test_submission = pd.read_csv('./input/gender_submission.csv')
test_submission = pd.read_csv('./input/target.csv')

datasets = [train,test]

# 统计name title的高频词
for df in datasets:
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)

# 高频词    
train.Title.value_counts(dropna=False)

# 高频词 与 生存 的关系， 比例
train.groupby('Title',as_index=False)['Survived'].mean().sort_values('Survived',ascending = False)
'''
           Title  Survived
    16       Sir  1.000000
    2   Countess  1.000000
    14        Ms  1.000000
    11       Mme  1.000000
    6       Lady  1.000000
    10      Mlle  1.000000
    13       Mrs  0.792000
    9       Miss  0.697802
    8     Master  0.575000
    1        Col  0.500000
    7      Major  0.500000
    4         Dr  0.428571
    12        Mr  0.156673
    5   Jonkheer  0.000000
    3        Don  0.000000
    15       Rev  0.000000
    0       Capt  0.000000
'''

# 清理数据：cabin, name, title, ticket  
for df in datasets:
    df['hasCabin'] = np.where(pd.isnull(df['Cabin']),0,1)
    df.loc[pd.isnull(df['Embarked']),'Embarked'] = 'None'
    df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
    
train.head()

# 处理 sex 和 embarked， 编码
SEED = 1
np.random.seed(SEED)
le = dict()
le['Sex'] = LabelEncoder()
le['Sex'].fit(train.Sex)
le['Embarked'] = LabelEncoder()
le['Embarked'].fit(train.Embarked)
le['Title'] = LabelEncoder()
le['Title'].fit(pd.concat([train.Title, test.Title], axis=0))

for df in datasets:
    df['Sex'] = le['Sex'].transform(df['Sex'])
    df['Embarked'] = le['Embarked'].transform(df['Embarked'])
    df['Title'] = le['Title'].transform(df['Title'])
    
train.head()

train.Title.value_counts(dropna=False)


# age 用平均值填空值
for df in datasets:
    df.loc[pd.isnull(df['Age']), 'Age'] = df['Age'].mean()

for df in datasets:
    df.loc[:,'Age'] = np.round(df['Age'])
            
# 票价
for df in datasets:
    df.loc[pd.isnull(df['Fare']),'Fare'] = df['Fare'].mean()


train.info()
test.info()

train.head()


# 热图
#plt.figure(figsize=(12,8))
#sns.heatmap(train.corr(), annot=True)
#plt.show()

# 数据整体描述
train.describe()

# 训练数据
x_train0 = train.drop(['PassengerId','Survived'],axis=1)
y_train0 = train['Survived']

# 测试数据
x_test0 = test.drop(['PassengerId'],axis=1)
y_test0 = test_submission['Survived']

# 数据准备

# 规范化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train0)
x_test = sc.fit_transform(x_test0)

y_train = y_train0.values.astype('float32')
y_test = y_test0.values.astype('float32')
