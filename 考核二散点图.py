from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

# 中文字体
font = {'family': 'Microsoft YaHei',
        'weight': 'bold'}

matplotlib.rc("font", **font)

# pandas读取csv的文件
data = pd.read_csv("D:\\桌面\\田字型散点.csv", header=None)
df = pd.DataFrame(data)

# 将数据转化为数组
df = np.array(df)

# 提取数据
x1 = df[0]
y1 = df[1]
l1 = df[2]

# 设置图形大小
plt.figure(figsize=(20, 8), dpi=80)

# 获取figure和axis
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# 隐藏上边和右边（框）
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 移动另外2个轴
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

# 设置x轴范围
plt.xlim((-10, 10))

# 调整x轴刻度
my_x_ticks = np.arange(-10, 12.5, 2.5)
plt.xticks(my_x_ticks)

# 标记0的点为蓝，1的为红
plt.scatter(x1[l1 == 0], y1[l1 == 0], c='blue')
plt.scatter(x1[l1 == 1], y1[l1 == 1], c='red')

# 图示
plt.legend()

# 绘图
plt.show()

