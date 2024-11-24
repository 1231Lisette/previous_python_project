import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import matplotlib
from sklearn.metrics import accuracy_score

# 字体
font = {'family': 'Microsoft YaHei',
        'weight': 'bold'}

matplotlib.rc("font", **font)

# 从CSV文件加载数据
data = pd.read_csv("D:\\桌面\\训练集.csv", header=None)

# 提取特征和标签
X = data.iloc[:2, :].values  # 前两行为特征
y = data.iloc[2, :].values   # 第三行为标签

# 创建线性核SVM模型
svm_model = svm.SVC(kernel='linear')

# 训练模型
svm_model.fit(X.T, y)

# 绘制决策边界和超平面
w = svm_model.coef_[0]  # 第一个类别的法向量的值
b = svm_model.intercept_[0]    # 截距（超平面与坐标轴的交点）

# 绘制数据点
plt.scatter(X[0], X[1], c=y, cmap=plt.cm.Paired)

# 绘制决策边界
xx = np.linspace(X[0].min(), X[0].max(), 10)    # 绘制超平面的x坐标
yy = - (w[0] * xx + b) / w[1]  # 根据超平面方程计算 yy
plt.plot(xx, yy, 'k-', label=f'Decision Boundary: {w[0]:.2f}x + {w[1]:.2f}y + {b:.2f} = 0')

# 绘制支持向量
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k')   # 支持向量的x、y坐标

# 将图例放在左上角
plt.legend(loc='upper left')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear SVM Classifier with Decision Boundary and Support Vectors')
plt.show()


# 从CSV文件加载测试集数据
test_data = pd.read_csv("D:\\桌面\\test.csv", header=None)

# 提取特征和标签
X_test = test_data.iloc[:2, :].values  # 前两行为特征
y_test = test_data.iloc[2, :].values   # 第三行为标签

# 在测试集上进行预测
y_pred = svm_model.predict(X_test.T)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)
