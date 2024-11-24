import pandas as pd
# 读取文件1：包含单品编码、单品名称和蔬菜种类
df1 = pd.read_excel("D:\\桌面\\附件1.xlsx")

# 读取文件2：包含年月日、商品编码、销售量、价格
df2 = pd.read_excel("D:\\桌面\\附件2.xlsx")

merged_df = pd.merge(df2, df1, left_on='单品编码', right_on='单品编码', how='left')

print(merged_df.head())

merged_df.to_excel("D:\\桌面\\merged.xlsx", index=False)
