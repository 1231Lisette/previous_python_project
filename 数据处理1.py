import pandas as pd

# 加载三个Excel文件为DataFrame
df1 = pd.read_excel("D:\\桌面\\summer_data.xlsx")  # 主要数据表
df2 = pd.read_excel("D:\\桌面\\附件4.xlsx")  # 包含单品编码和损耗率
df3 = pd.read_excel("D:\\桌面\\附件3.xlsx")  # 包含date、单品编码和批发价

print("df2 columns:", df2.columns)
print("df3 columns:", df3.columns)
