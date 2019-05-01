# Gait Recognition
data文件下存放的是加速度、陀螺仪数据，均为3维数据；

util获取初始数据、以及数据的后续处理

model下是rnn、cnn模型，包括网络结构及各项参数。rnn与cnn的训练、测试共用一份代码。
## demo
包括PCA-KNN、KNN、SVM
## cnn
(Conv1 + Pool1) x 3
## rnn
lstm、gru可选
