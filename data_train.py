#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models
from keras import layers

from data_prepare import *

# 三层网络 模型定义
model = models.Sequential()
#model.add(layers.Dense(4, activation='tanh', input_shape=(8,)))
#model.add(layers.Dense(4, activation='tanh' ))
#model.add(layers.Dense(4, activation='tanh' ))
#model.add(layers.Dense(1, activation='sigmoid'))

model.add(layers.Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 8))
model.add(layers.Dense(output_dim = 2, init = 'uniform', activation = 'relu'))
model.add(layers.Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#训练模型
history = model.fit(x_train, y_train, epochs=100, batch_size=10)

# 评估预测结果
results = model.evaluate(x_test, y_test)
print(results)

# 用模型进行评估，输出预测结果集
predict = model.predict(x_test)


my_submission = pd.DataFrame({
	'PassengerId': test.PassengerId, 
	'Survived': pd.Series(predict.reshape((1,-1))[0]).round().astype(int)
})

my_submission.head()

my_submission.to_csv('submission.csv', index=False)


match = 0
nomatch = 0
predict2 = predict.round() 
for x in range(len(y_test)):
    if predict2[x][0] == y_test[x][0]:
        match = match +1
    else:
        nomatch = nomatch +1

print('match=', match)
print('nomatch=', nomatch)
print(match/(match+nomatch))

