#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models
from keras import layers

from data_prepare import *

# 三层网络 模型定义
model = models.Sequential()
model.add(layers.Dense(4, activation='tanh', input_shape=(8,)))
model.add(layers.Dense(4, activation='tanh' ))
model.add(layers.Dense(4, activation='tanh' ))
model.add(layers.Dense(1, activation='sigmoid'))

#编译模型
model.compile(optimizer='rmsprop',
			  loss='mse',
			  #loss='binary_crossentropy',
			  metrics=['acc']
			  )

#留出验证集
x_val = x_train[:100]
partial_x_train = x_train[100:]
y_val = y_train[:100]
partial_y_train = y_train[100:]


#训练模型
history = model.fit(partial_x_train,
		  partial_y_train,
		  epochs=100,
		  batch_size=4,
		  validation_data=(x_val, y_val))

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

