import pandas as pd

# 读取数据
t2 = pd.read_excel("cleaned_data_with_sale_volume.xlsx")
att2 = pd.read_excel("附件2.xlsx", sheet_name=["2023年统计的相关数据"])

# 获取 '2023年统计的相关数据' 表格
att2_data = att2["2023年统计的相关数据"]

# 根据条件合并数据
t2.loc[
    (t2['crop_name'].isin(att2_data['作物名称'])) & (t2['plot_type'].isin(att2_data['地块类型'])),
    'crop_name'
] = t2.apply(
    lambda row: att2_data.loc[
        (att2_data['作物名称'] == row['crop_name']) &
        (att2_data['地块类型'] == row['plot_type']),
        '作物编号'
    ].values[0] if not att2_data.loc[
        (att2_data['作物名称'] == row['crop_name']) &
        (att2_data['地块类型'] == row['plot_type'])
    ].empty else row['crop_name'],
    axis=1
)

# 保存修改后的 DataFrame
t2.to_excel("1.xlsx", index=False)
