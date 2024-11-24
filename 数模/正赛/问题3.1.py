import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 设置中文字体
font = {'family': 'Microsoft YaHei', 'weight': 'bold'}
matplotlib.rc("font", **font)

# 读取 Excel 文件，设置第一列和第一行为索引和列名称
complementarity_matrix = pd.read_excel("D:\\桌面\\替代性系数.xlsx", index_col=0, header=0)

# 分割成四个 10x10 子矩阵和一个 11x11 子矩阵
matrix_10x10_1 = complementarity_matrix.iloc[:10, :10]
matrix_10x10_2 = complementarity_matrix.iloc[:10, 10:20]
matrix_10x10_3 = complementarity_matrix.iloc[10:20, :10]
matrix_10x10_4 = complementarity_matrix.iloc[10:20, 10:20]
matrix_11x11 = complementarity_matrix.iloc[20:, 20:]

# 打印子矩阵数据，检查分割是否正确
# print("matrix_10x10_1:\n", matrix_10x10_1)
# print("matrix_10x10_2:\n", matrix_10x10_2)
# print("matrix_10x10_3:\n", matrix_10x10_3)
# print("matrix_10x10_4:\n", matrix_10x10_4)
# print("matrix_11x11:\n", matrix_11x11)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(matrix_10x10_1, annot=True, cmap='YlGnBu', square=True, cbar_kws={'shrink': .8})
plt.title('替代性系数矩阵 (10x10 - Part 1)')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(matrix_10x10_2, annot=True, cmap='YlGnBu', square=True, cbar_kws={'shrink': .8})
plt.title('替代性系数矩阵 (10x10 - Part 2)')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(matrix_10x10_3, annot=True, cmap='YlGnBu', square=True, cbar_kws={'shrink': .8})
plt.title('替代性系数矩阵 (10x10 - Part 3)')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(matrix_10x10_4, annot=True, cmap='YlGnBu', square=True, cbar_kws={'shrink': .8})
plt.title('替代性系数矩阵 (10x10 - Part 4)')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(matrix_11x11, annot=True, cmap='YlGnBu', square=True, cbar_kws={'shrink': .8})
plt.title('替代性系数矩阵 (11x11 - Part 5)')
plt.show()