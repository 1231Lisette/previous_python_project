import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

# 从CSV文件加载训练集数据
train_data = pd.read_csv("D:\\桌面\\训练集.csv", header=None)

# 提取特征和标签
X_train = train_data.iloc[:2, :].values  # 前两行为特征
y_train = train_data.iloc[2, :].values   # 第三行为标签

# 转置特征向量，确保每列代表一个样本
X_train = X_train.T

# 创建SVM模型
svm_model = svm.SVC(kernel='rbf')

# 在训练集上训练模型
svm_model.fit(X_train, y_train)

# 创建网格以绘制决策边界
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和超平面
plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)

# 绘制支持向量
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k')

# 导入测试集数据
test_data = pd.read_csv("D:\\桌面\\test.csv", header=None)

# 提取特征和标签
X_test = test_data.iloc[:2, :].values  # 前两行为特征
y_test = test_data.iloc[2, :].values   # 第三行为标签

# 转置特征向量，确保每列代表一个样本
X_test = X_test.T

# 在测试集上进行预测
y_pred = svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)
