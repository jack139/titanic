#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models
from keras import layers

from data_prepare import *

# 三层网络 模型定义
model = models.Sequential()
model.add(layers.Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
model.add(layers.Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
model.add(layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#训练模型
history = model.fit(x_train, y_train, epochs=50, batch_size=8)

# 评估预测结果
results = model.evaluate(x_test, y_test)
print(results)

# 用模型进行评估，输出预测结果集
predict = model.predict(x_test)


# 生成并保存提交文件
my_submission = pd.DataFrame({
	'PassengerId': test.PassengerId, 
	'Survived': pd.Series(predict.reshape((1,-1))[0]).round().astype(int)
})

my_submission.head()

my_submission.to_csv('submission.csv', index=False)


# 计算正确率
match = 0
nomatch = 0
predict2 = predict.round() 
for x in range(len(y_test)):
    if predict2[x][0] == y_test[x]:
        match = match +1
    else:
        nomatch = nomatch +1

print('match=', match)
print('nomatch=', nomatch)
print(match/(match+nomatch))

