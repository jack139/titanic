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
train.head()

test = pd.read_csv('./input/test.csv')
test.head()

test_submission = pd.read_csv('./input/gender_submission.csv')
test_submission.head()

train.info()
test.info()
test_submission.info()

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
    df['TitleEnc'] = le['Title'].transform(df['Title'])
    
train.head()

train.TitleEnc.value_counts(dropna=False)

# 统计 family， 有 伴侣或家属
for df in datasets:
    df['Family'] = np.where((df['SibSp'] > 2),0,np.where((df['Parch'] < 2),2,1))
    
print('Sobreviventes:\n',train.groupby(['Family'])['Survived'].mean())

# title, family, age 关系
titles = train.TitleEnc.unique()
family = train.Family.unique()
titles.sort()
family.sort()

for f in family:
    for title in titles:
        for df in datasets:
            df.loc[(pd.isnull(df['Age'])) & 
                   (df['TitleEnc'] == title) &
                   (df['Family'] == f), 'Age'] = df[(df['TitleEnc'] == title) & (df['Family'] == f)]['Age'].mean()

for df in datasets:
    df.loc[:,'Age'] = np.round(df['Age'])
            
train.info()


# 票价
for df in datasets:
    df.loc[pd.isnull(df['Fare']),'Fare'] = df['Fare'].mean()

# 热图
#plt.figure(figsize=(12,8))
#sns.heatmap(train.corr(), annot=True)
#plt.show()


# 年龄
def ageGroup(age):
    if age < 5 or age > 79:
        return 1
    if age < 18:
        return 2
    if age < 60:
        return 3
    return 4

for df in datasets:
    df['AgeGroup'] = df['Age'].apply(ageGroup)
    
print('Sobreviventes:\n',train.groupby(['AgeGroup'])['Survived'].mean())

# 综合属性，---- 运行有错误
for df in datasets:
    df['Preference'] = np.where(df['TitleEnc']>2, 10, 2) # 有错误，where需要补全
    df['Preference'] = df['Preference'] * df['Sex']
    df['Preference'] = df['Preference'] // df['Pclass']
    df['Preference'] = df['Preference'] // df['AgeGroup']

print('Sobreviventes:\n',train.groupby(['Preference'])['Survived'].mean())

# 清除 无用的列
for df in datasets:
    df.drop(['Title', 'Age','SibSp','Parch'],axis=1,inplace=True)
    
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


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train0)
x_test = sc.fit_transform(x_test0)

# 数据准备
y_train = y_train0.values.astype('float32')
y_test = y_test0.values.astype('float32')
