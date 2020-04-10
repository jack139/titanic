#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras import optimizers

from data_prepare4 import *

# 模型参数
epochs_num = 15
batch_size = 100
input_dim = x_train.shape[1]

print('input_dim=', input_dim, ' batch_size=', batch_size, ' epochs_num=', epochs_num)

# 创建模型
def get_model(input_dim):
    # 三层网络 模型定义
    model = models.Sequential()
    model.add(layers.Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
    model.add(layers.Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(layers.Dense(units = 2, kernel_initializer = 'uniform', activation = 'softmax'))
    #编译模型
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy', metrics=['accuracy']) 

    #model = Sequential()
    #model.add(layers.Dense(units=50, input_dim=input_dim, kernel_initializer='normal', bias_initializer='zeros'))
    #model.add(layers.Activation('relu'))
    #for i in range(0, 5):
    #    model.add(layers.Dense(units=20, kernel_initializer='normal', bias_initializer='zeros'))
    #    model.add(layers.Activation('relu'))
    #    model.add(layers.Dropout(.40))
    #model.add(layers.Dense(units=2))
    #model.add(layers.Activation('softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
history = model.fit(x_train, y_train, epochs=epochs_num, batch_size=batch_size, verbose=1,
    validation_data=(x_test, y_test))

# 评估预测结果
results = model.evaluate(x_test, y_test, verbose=1)
print('predict: ', results)

# 用模型进行评估，输出预测结果集
predict = model.predict_classes(x_test)


# 生成并保存提交文件
my_submission = pd.DataFrame({
    'PassengerId': test_data.index, 
    'Survived': predict
})

my_submission.head()

my_submission.to_csv('submission.csv', index=False)

# 计算正确率
#match = 0
#nomatch = 0
#predict2 = predict.round() 
#for x in range(len(y_test)):
#    if predict2[x][0] == y_test[x]:
#        match = match +1
#    else:
#        nomatch = nomatch +1
#
#print('match=', match)
#print('nomatch=', nomatch)
#print(match/(match+nomatch))


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, acc, 'go', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
