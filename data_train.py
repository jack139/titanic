#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras import optimizers

from data_prepare3 import *

# 模型参数
epochs_num = 32
batch_size = 5
input_dim = len(x_train[0])

print('input_dim=', input_dim, ' batch_size=', batch_size, ' epochs_num=', epochs_num)

# 创建模型
def get_model(input_dim):
    # 三层网络 模型定义
    model = models.Sequential()
    model.add(layers.Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
    model.add(layers.Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    #编译模型
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy', metrics=['accuracy']) 
    return model

#np.random.shuffle(data)

## K 折交叉验证
#k = 3
#num_validation_samples = len(x_train) // k
#
#validation_scores = []
#
#for fold in range(k):
#    print('processing fold #', fold)
#    #选择验证数据分区
#    validation_x = x_train[num_validation_samples * fold:num_validation_samples * (fold + 1)]
#    validation_y = y_train[num_validation_samples * fold:num_validation_samples * (fold + 1)]
#    #使用剩余数据作为训练数据
#    training_x = np.concatenate([x_train[:num_validation_samples * fold],
#        x_train[num_validation_samples * (fold + 1):]], axis=0)
#    training_y = np.concatenate([y_train[:num_validation_samples * fold],
#        y_train[num_validation_samples * (fold + 1):]], axis=0)
#
#    #创建一个全新的模型实例（未训练）
#    model = get_model(input_dim)
#
#    #训练模型
#    history = model.fit(training_x, training_y, epochs=epochs_num, batch_size=batch_size, verbose=0)
#
#    # 验证
#    validation_score = model.evaluate(validation_x, validation_y, verbose=0)
#    validation_scores.append(validation_score)
#    print(validation_score)
#
## 最终验证分数，K折验证的平均值
#validation_score = np.average(validation_scores)
#print('avg: ', validation_score)

##创建一个全新的模型实例（未训练）
model = get_model(input_dim)

#训练模型
history = model.fit(x_train, y_train, epochs=epochs_num, batch_size=batch_size, verbose=1)

# 评估预测结果
results = model.evaluate(x_test, y_test, verbose=1)
print('predict: ', results)

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

