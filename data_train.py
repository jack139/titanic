#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models
from keras import layers

from data_prepare import *

# 三层网络 模型定义
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(8,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#编译模型
model.compile(optimizer='rmsprop',
			  loss='binary_crossentropy',
			  #loss='mse',
			  metrics=['acc']
			  )

#留出验证集
x_val = x_train[:50]
partial_x_train = x_train[50:]
y_val = y_train[:50]
partial_y_train = y_train[50:]


#训练模型
model.fit(partial_x_train,
		  partial_y_train,
		  epochs=30,
		  batch_size=8,
		  validation_data=(x_val, y_val))


results = model.evaluate(x_test, y_test)
print(results)

