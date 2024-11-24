import pandas as pd

# 假设 pivot_table 是你已经生成好的透视表
# 读取透视表数据
pivot_table = pd.read_excel("D:\\桌面\\top_bottom_10_sorted_pivot_table.xlsx", index_col=0)

# 将date索引转换为 "YYYYSeason" 格式
def date_to_season(date):
    year = date.year
    month = date.month
    if month in [12, 1, 2]:
        season = 'Winter'
    elif month in [3, 4, 5]:
        season = 'Spring'
    elif month in [6, 7, 8]:
        season = 'Summer'
    else:
        season = 'Autumn'
    return f"{year}{season}"

# 更新透视表的索引
pivot_table.index = pivot_table.index.map(pd.to_datetime).map(date_to_season)

# 对相同的 "YYYYSeason" 进行汇总
pivot_table = pivot_table.groupby(pivot_table.index).sum()

# 打印结果或保存为Excel
print("Updated Pivot Table with Date in 'YYYYSeason' Format:")
print(pivot_table)

# 保存为Excel文件
output_file = "D:\\桌面\\updated_pivot_table_with_season.xlsx"
pivot_table.to_excel(output_file)
