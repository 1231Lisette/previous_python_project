import pandas as pd

# 假设 df 是从 Excel 读取的数df = pd.read_excel("your_file_path.xlsx")

# 示例数据结构
# df = pd.DataFrame({
#     'date': ['2024-08-01', '2024-08-01', '2024-08-02', ...],
#     'category': ['花叶', '花叶', '辣椒', ...],
#     'single_name': ['小类1', '小类2', '小类1', ...],
#     'sales_kg': [100, 200, 150, ...]
# })

# 将日期列转换为datetime格式
df = pd.read_excel("D:\\桌面\\merged.xlsx")

# 计算每个菜品小类的总销量
total_sales = df.groupby('single_name')['sales_kg'].sum()

# 找到总销量前10的菜品
top_10 = total_sales.nlargest(10).index

# 找到总销量后10的菜品
bottom_10 = total_sales.nsmallest(10).index

# 合并前10和后10的菜品
combined_top_bottom = top_10.union(bottom_10)

# 筛选出这些菜品的数据
combined_df = df[df['single_name'].isin(combined_top_bottom)]

# 创建透视表
pivot_table = combined_df.pivot_table(
    index='date',
    columns='single_name',
    values='sales_kg',
    aggfunc='sum',
    fill_value=0
)

# 按照总销量降序排列列顺序
sorted_columns = total_sales.loc[combined_top_bottom].sort_values(ascending=False).index
pivot_table = pivot_table[sorted_columns]

# 打印结果或保存为Excel
print("Pivot Table for Top 10 and Bottom 10 (Sorted by Total Sales):")
print(pivot_table)

# 保存为Excel文件
output_file = "D:\\桌面\\top_bottom_10_sorted_pivot_table.xlsx"
pivot_table.to_excel(output_file)
