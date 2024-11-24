import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 加载训练集数据
train_data = pd.read_csv("D:\\桌面\\训练集.csv", header=None)

# 提取特征和标签
X_train = train_data.iloc[:2, :].values  # 前两行为特征
y_train = train_data.iloc[2, :].values   # 第三行为标签

# 创建并训练逻辑回归模型
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train.T, y_train)

# 加载测试集数据
test_data = pd.read_csv("D:\\桌面\\test.csv", header=None)

# 提取特征和标签
X_test = test_data.iloc[:2, :].values  # 前两行为特征
y_test = test_data.iloc[2, :].values   # 第三行为标签

# 使用训练好的模型进行预测
y_pred = log_reg_model.predict(X_test.T)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)

# 设置图的大小
plt.figure(figsize=(8, 6))

# 绘制决策边界和数据点
scatter_train = plt.scatter(X_train[0], X_train[1], c=y_train, cmap=plt.cm.Paired)
scatter_test = plt.scatter(X_test[0], X_test[1], c=y_pred, marker='x', cmap=plt.cm.Paired)  # 测试集点用叉号表示

# 绘制决策边界
w1, w2 = log_reg_model.coef_[0]
b = log_reg_model.intercept_[0]
x_boundary = np.array([X_train[0].min(), X_train[0].max()])
y_boundary = -(x_boundary * w1 + b) / w2
line_boundary, = plt.plot(x_boundary, y_boundary, 'k-', label=f'Decision Boundary: {w1:.2f}x + {w2:.2f}y + {b:.2f} = 0')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Classifier with Decision Boundary')

# 设置图例的位置、大小和透明度
plt.legend((scatter_train, scatter_test, line_boundary), ('Train Points', 'Test Points', 'Decision Boundary'),
           loc='upper left', fontsize='large')
plt.show()