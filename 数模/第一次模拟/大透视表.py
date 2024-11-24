import pandas as pd

# 假设 df 是从 Excel 读取的数据
df = pd.read_excel("D:\\桌面\\merged.xlsx")

# 示例数据结构
# df = pd.DataFrame({
#     'date': ['2024-08-01', '2024-08-01', '2024-08-02', ...],
#     'category': ['花叶', '花叶', '辣椒', ...],
#     'single_name': ['小类1', '小类2', '小类1', ...],
#     'sales_kg': [100, 200, 150, ...]
# })

# 将日期列转换为datetime格式
df['date'] = pd.to_datetime(df['date'])

# 定义6个蔬菜大类
categories = ['花叶类', '花菜类', '辣椒类', '茄类', '水生根茎类', '食用菌']

# 创建一个字典来保存每个大类的透视表
pivot_tables = {}

# 循环生成每个大类的透视表
for category in categories:
    # 筛选出当前大类的数据
    category_df = df[df['category'] == category]

    # 创建透视表
    pivot_table = category_df.pivot_table(
        index='date',
        columns='single_name',
        values='sales_kg',
        aggfunc='sum',
        fill_value=0
    )

    # 将透视表存入字典中
    pivot_tables[category] = pivot_table

    # 打印结果或保存为Excel
    print(f"Category: {category}")
    print(pivot_table)
    print("\n")

    # 保存为Excel文件
    output_file = f"D:\\桌面\\{category}_pivot_table.xlsx"
    pivot_table.to_excel(output_file)
