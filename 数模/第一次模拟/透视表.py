import pandas as pd

# 1. 读取 Excel 文件
file_path = 'D:\\桌面\\hc.xlsx'  # 请根据实际文件路径修改
df = pd.read_excel(file_path)

# 2. 提取表头的后三列
# 假设列名为 'single_name', 'sales_kg', 'year&season'
# 可以先查看一下实际的列名
print("列名列表：", df.columns)

# 提取后三列
df_selected = df[['single_name', 'sales_kg', 'date']]

# 3. 生成透视表
# 透视表中，'year&season' 为行索引，'single_name' 为列索引，'sales_kg' 为值
pivot_table = df_selected.pivot_table(
    index='date',    # 行索引
    columns='single_name',  # 列索引
    values='sales_kg',      # 值
    aggfunc='sum',          # 聚合函数，这里是求和
    fill_value=0            # 填充缺失值为0
)

# 4. 输出透视表
print(pivot_table)

# 如果需要，将透视表保存为新的 Excel 文件
pivot_table.to_excel('D:\\桌面\\pivot_hchc.xlsx')
