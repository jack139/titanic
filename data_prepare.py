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

# 建分类信息，可通过title进行分类query
def encTitle(title):
    title = str(title)
    if title in ('Mr'):
        return 1
    elif title in ('Miss','Mrs'):
        return 2
    elif title in ('Master','Sir'):
        return 3
    elif title in ('Rev','Jonkheer','Don','Countess','Col'):
        return 4
    elif title in ('Dr'):
        return 5
    elif title in ('Major','Capt'):
        return 6
    elif title in ('Lady','Mlle','Don'):
        return 7
    elif title in ('Mme','Ms'):
        return 8
    else:
        return 9
    
def encTitle(title):
    title = str(title)
    if title in ('Jonkheer','Don','Rev','Capt'):
        return 1
    elif title in ('Mr'):
        return 2
    elif title in ('Dr','Major','Col','Master'):
        return 3
    elif title in ('Miss','Mrs'):
        return 4
    elif title in ('Ms','Mme','Sir'):
        return 5
    elif title in ('Mlle','Lady','Countess'):
        return 6

for df in datasets:
    df['TitleEnc'] = df['Title'].apply(encTitle)
    
train.TitleEnc.value_counts(dropna=False)

train.query("TitleEnc == 7")


# 清理数据：cabin, name, title, ticket  
for df in datasets:
    df['hasCabin'] = np.where(pd.isnull(df['Cabin']),0,1)
    df.loc[pd.isnull(df['Embarked']),'Embarked'] = 'None'
    df.drop(['Title','Name','Ticket','Cabin'],axis=1,inplace=True)
    
train.head()

# 处理 sex 和 embarked
SEED = 1
np.random.seed(SEED)
le = dict()
le['Sex'] = LabelEncoder()
le['Sex'].fit(train.Sex)
le['Embarked'] = LabelEncoder()
le['Embarked'].fit(train.Embarked)

for df in datasets:
    df['Sex'] = le['Sex'].transform(df['Sex'])
    df['Embarked'] = le['Embarked'].transform(df['Embarked'])
    
train.head()

# 统计 family， 有 伴侣或家属
for df in datasets:
    df['Family'] = np.where((df['SibSp'] > 1),0,np.where((df['Parch'] > 1),2,1))
    
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

# title 生存概率
print('Sobreviventes:\n',train.groupby(['TitleEnc'])['Survived'].mean())

# 调整 title 参数
def newTitleEnc(n):
    if n in (1,4,6):
        return 1
    if n in (2,3,5):
        return 2
    if n in (7,8):
        return 3
    return 1

for df in datasets:
    df.loc[:,'TitleEnc'] = df['TitleEnc'].apply(newTitleEnc)

print('Sobreviventes:\n',train.groupby(['TitleEnc'])['Survived'].mean())

# 年龄
def ageGroup(age):
    if age < 18:
        return 1
    if age < 60:
        return 2
    return 3

for df in datasets:
    df['AgeGroup'] = df['Age'].apply(ageGroup)
    
print('Sobreviventes:\n',train.groupby(['AgeGroup'])['Survived'].mean())

# 综合属性，---- 运行有错误
#for df in datasets:
#    df['Preference'] = np.where(df['TitleEnc']) # 有错误，where需要补全
#    df['Preference'] = df['Preference'] * df['Sex']
#    df['Preference'] = df['Preference'] // df['Pclass']
#    df['Preference'] = df['Preference'] // df['AgeGroup']
#
#print('Sobreviventes:\n',train.groupby(['Preference'])['Survived'].mean())

# 清除 无用的列
for df in datasets:
    df.drop(['Age','SibSp','Parch'],axis=1,inplace=True)
    
train.head()


# 热图
#plt.figure(figsize=(12,8))
#sns.heatmap(train.corr(), annot=True)
#plt.show()

# 数据整体描述
train.describe()

# 训练数据
x_train = train.drop(['PassengerId','Survived'],axis=1)
y_train = train['Survived']

# 测试数据
x_test = test.drop(['PassengerId'],axis=1)
y_test = test_submission.drop(['PassengerId'],axis=1)
