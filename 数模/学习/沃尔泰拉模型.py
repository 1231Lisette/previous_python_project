from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def volterra_model(X, t, alpha, beta, delta, gamma, c):
    P, V = X
    dPdt = alpha * P - beta * P * V - c * P
    dVdt = delta * P * V - gamma * V
    return [dPdt, dVdt]

P0 = 100    # 初始猎物数量
V0 = 20     # 初始捕食者数量
t = np.linspace(0, 10, 100)     # 时间范围
alpha = 1.0     # 猎物增长率
beta = 0.1  # 捕食者对猎物的影响系数
delta = 0.1     # 猎物对捕食者的影响系数
gamma = 1.0     # 捕食者死亡率
c = 0.05    # 捕获强度

# 求解微分方程
X0 = [P0, V0]   # 初始状态
sol = odeint(volterra_model, X0, t, args=(alpha, beta, delta, gamma, c))
P, V = sol[:, 0], sol[:, 1]

# 绘制图像
plt.plot(t,P, label= 'prey')
plt.plot(t,V, label= 'predator')
plt.xlabel('time')
plt.ylabel('population')
plt.title('Volterra Model with Capture Intensity')
plt.legend()
plt.show()

