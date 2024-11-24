import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 字体设置
font = {'family': 'Microsoft YaHei',
        'weight': 'bold'}

matplotlib.rc("font", **font)

# 假设我们有一组数据点
v = np.array([1.913, 2.406, 2.892, 3.337, 3.827, 4.325])    # 横坐标
i = np.array([2.00, 2.50, 3.00, 3.50, 4.00, 4.50])    # 纵坐标

# 构建增广矩阵，将 x 变量转换为 [x+1] 形式
A = np.vstack([v, np.ones(len(v))]).T

# 使用最小二乘法进行拟合
m, c = np.linalg.lstsq(A, i, rcond=None)[0]

# 输出拟合的直线参数
print("斜率 m:", m)
print("截距 c:", c)

# 设置 x 轴和 y 轴的刻度数
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)

# 设置 x 轴范围
plt.xlim(0, 5)

plt.grid(alpha=0.5)

# 绘制原始数据点和拟合的直线，并添加图例
plt.plot(v, i, 'o', label='原始数据', markersize=8)
plt.plot(v, m*v + c, 'r', label='拟合直线 (y = {:.4f}x + {:.4f})'.format(m, c))
plt.legend()
plt.show()
