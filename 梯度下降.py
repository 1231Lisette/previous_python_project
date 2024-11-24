import torch
# 实现梯度下降的算法
# 预测模型： f(x) = mx+b，用于预测房屋的真实价格

import numpy as np
import matplotlib.pyplot as plt

data = np.array([[80, 200],
                 [95, 230],
                 [104, 245],
                 [112, 247],
                 [125, 259],
                 [135, 262]])

# 求解f(x) = mx + b  ，其中（x，y）来自data，y为标记数据
# 目标：y 与 f(x)之间的差距尽量小

# 初始化参数
m = 1
b = 1
lr = 0.00001


#   梯度下降的函数
def gradientdecent(m, b, data, lr):  # 当前的m，b和数据data，学习率lr

    loss, mpd, bpd = 0, 0, 0  # loss 为均方误差，mpd为m的偏导数，
    # bpd为b的偏导数
    for xi, yi in data:
        loss += (m * xi + b - yi) ** 2  # 计算mse
        bpd += (m * xi + b - yi) * 2  # 计算loss/b偏导数
        mpd += (m * xi + b - yi) * 2 * xi  # 计算loss/m偏导数


# 更新m,b
    N = len(data)
    loss = loss / N
    mpd = mpd / N
    bpd = bpd / N
    m = m - mpd * lr
    b = b - bpd * lr
    return loss, m, b

# 训练过程，循环一定的次数，或者符合某种条件后结束
for epoch in range(30000000):
    mse,m ,b = gradientdecent(m,b,data,lr)
    if epoch% 100000 == 0:
        print(f"loss={mse:.4f},m={m:.4f},b={b:.4f}")


