import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
file_path = "D:\\桌面\\pivot_table.xlsx"  # 更新为实际文件路径
df = pd.read_excel(file_path)

# 去掉空值，防止PCA计算时出错
df = df.dropna()

# 去掉year&season列，只保留菜品数据进行分析
df_pca = df.drop(columns=['year&season'])

# 标准化数据
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca)

# 进行PCA分析
pca = PCA(n_components=2)  # 将数据降到2个维度
pca_result = pca.fit_transform(df_scaled)

# 将PCA结果转换为DataFrame并加入菜品名称
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=df_pca.columns)

# 绘制PCA结果
plt.figure(figsize=(10, 7))
plt.scatter(pca_df['PC1'], pca_df['PC2'])

# 添加每个菜品名称为标注
for i, txt in enumerate(pca_df.index):
    plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=8)

plt.title('PCA of Vegetable Sales Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
