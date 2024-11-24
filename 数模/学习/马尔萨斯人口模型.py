from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# 定义马尔萨斯人口模型的微分方程
def malthusian_model(P, t, r):
    dPdt = r * P
    return dPdt

# 定义初始条件和参数
P0 = 100    #初始人口数量
t = np.linspace(0, 1000, 1000)  #时间范围
r = 0.01  #人口增长率,不可能为常数

# 求解微分方程
P = odeint(malthusian_model, P0, t, args=(r,))

# 绘制人口随时间变化的图像
plt.plot(t,P)
plt.xlabel('time')
plt.ylabel('population')
plt.title('Malthusian Population Model')
plt.show()
