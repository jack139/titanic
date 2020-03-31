#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense

from data_prepare2 import *

classifier = Sequential()

#Input layer with 5 inputs neurons
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))
#Hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))
#output layer with 1 output neuron which will predict 1 or 0
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


#compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#train
classifier.fit(X_train, y_train, batch_size = 4, nb_epoch = 100)

#getting predictions of test data
prediction = classifier.predict(X_test).tolist()
# list to series
se = pd.Series(prediction)
# creating new column of predictions in data_check dataframe
data_check['check'] = se
data_check['check'] = data_check['check'].str.get(0)


series = []
for val in data_check.check:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)
data_check['final'] = series


match = 0
nomatch = 0
for val in data_check.values:
    if val[1] == val[3]:
        match = match +1
    else:
        nomatch = nomatch +1

print('match=', match)
print('nomatch=', nomatch)
print(match/(match+nomatch))




# 用模型进行评估，输出预测结果集
#predict = model.predict(x_test)
#
#my_submission = pd.DataFrame({
#	'PassengerId': test.PassengerId, 
#	'Survived': pd.Series(predict.reshape((1,-1))[0]).round().astype(int)
#})
#
#my_submission.head()
#
#my_submission.to_csv('submission.csv', index=False)

