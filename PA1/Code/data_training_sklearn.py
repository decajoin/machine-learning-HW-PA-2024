from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# 读取数据
df = pd.read_csv('./Data/advertising.csv')

# 预览数据
print(df.head())

# 'sales' 是目标变量，其余是特征
X = df.drop('sales', axis=1)  # 特征
y = df['sales']  # 目标变量

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 打印划分后的数据集大小
print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在训练集上进行预测
y_train_pred = model.predict(X_train)

# 在测试集上进行预测
y_test_pred = model.predict(X_test)

# 计算训练集上的预测误差
mse_train = mean_squared_error(y_train, y_train_pred)
print("训练集上的预测误差（MSE）:", mse_train)

# 计算测试集上的预测误差
mse_test = mean_squared_error(y_test, y_test_pred)
print("测试集上的预测误差（MSE）:", mse_test)

# 导出训练得到的参数
weights = model.coef_
bias = model.intercept_

# 打印参数
print("权重:", weights)
print("偏置项:", bias)
