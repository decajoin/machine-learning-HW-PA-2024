from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('./Data/advertising.csv')

# 'sales' 是目标变量，其余是特征
X = df.drop('sales', axis=1)  # 特征
y = df['sales']  # 目标变量

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 打印划分后的数据集大小
print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 添加偏置项
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# 预览 X_train 数据
print(X_train[:5])

# 初始化参数
theta = np.zeros(X_train.shape[1]) * 0.01

# 梯度下降参数
alpha = 0.000001  # 学习速率
iterations = 1000 # 迭代次数

# 存储损失函数值
losses = []

# 存储测试集的损失函数值
test_losses = []

# 梯度下降算法
for i in range(iterations):
    # 预测值
    y_pred = np.dot(X_train, theta)
    # 损失函数（均方误差）
    loss = np.mean((y_pred - y_train) ** 2)
    losses.append(loss)

    # 计算测试集的损失函数值
    test_y_pred = np.dot(X_test, theta)
    test_loss = np.mean((test_y_pred - y_test) ** 2)
    test_losses.append(test_loss)

    # 梯度
    gradient = np.dot(X_train.T, (y_pred - y_train)) / X_train.shape[0]

    # 更新参数
    theta -= alpha * gradient

# 将迭代次数和损失值保存为DataFrame
loss_df = pd.DataFrame({
    'iteration': range(1, iterations + 1),
    'loss': losses
})

# 将迭代次数和测试集损失值保存为DataFrame
test_loss_df = pd.DataFrame({
    'iteration': range(1, iterations + 1),
    'test_loss': test_losses
})

# 保存为CSV文件
loss_df.to_csv('./Data/losses.csv', index=False)
test_loss_df.to_csv('./Data/test_losses.csv', index=False)

# 输出最后损失值
print('训练集损失值：', loss_df.iloc[-1]['loss'])
print('测试集损失值：', test_loss_df.iloc[-1]['test_loss'])

# 保存参数
np.savetxt('./Data/theta.csv', theta, delimiter=',')
print('训练完成！')