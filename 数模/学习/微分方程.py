import numpy as np
from scipy.integrate import odeint

# 微分方程函数
def model(y, t):
    k = 0.3
    dydt = -k * y
    return dydt

# 初始条件
y0 = 5

# 时间点
t = np.linspace(0, 20, 100)

# 求解ODE
result = odeint(model, y0, t)

# 输出结果
print(result)