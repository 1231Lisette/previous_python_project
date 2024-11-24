import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 中文字体
font = {'family': 'Microsoft YaHei',
        'weight': 'bold'}

matplotlib.rc("font", **font)

data = pd.read_excel("D:\\桌面\\指标之间.xlsx")
data_subset = data[['销售价格', '种植成本', '预期销售量']]

corr_matrix = data_subset.corr(method='pearson')
print(corr_matrix)

# 可视化相关性矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Crop Data')
plt.show()