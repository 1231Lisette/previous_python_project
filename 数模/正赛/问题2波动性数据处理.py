import pandas as pd
import numpy as np


cost_growth = 0.05  # 成本增长率（5%每年）
price_decline_yangdujun = -0.05  # 羊肚菌价格下滑（-5%）

file_path = "D:\\桌面\\att2.xlsx"  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 定义未来的年份范围
years = list(range(2024, 2031))

# 定义无效的作物和地类型组合
invalid_combinations = [
    ('大白菜', '普通大棚'),
    ('玉米', '智慧大棚')
]

# 创建存储模拟数据的字典
simulated_data = {
    '序号': [],
    '作物编号': [],
    '作物名称': [],
    '地块类型': [],
    '种植季次': [],
    '亩产量/斤': [],
    '种植成本/(元/亩)': [],
    '销售单价/(元/斤)': [],
    '最小销售单价': [],
    '最大销售单价': [],
    '平均销售单价': [],
    '种植面积/亩': [],
    '生产量/斤': [],
    '年份': []  # 添加年份列
}

# 遍历每种作物和年份生成模拟数据
for idx, row in data.iterrows():
    crop_name = row['作物名称']
    crop_num = row['作物编号']
    avg_sale_price = row['平均销售单价']
    min_sale_price = row['最小销售单价']
    max_sale_price = row['最大销售单价']
    yield_per_acre = row['亩产量/斤']
    plant_cost = row['种植成本/(元/亩)']
    land_type = row['地块类型']
    season_type = row['种植季次']
    plant_area = row['种植面积/亩']

    # 跳过不允许的作物和地类型组合
    if (crop_name, land_type) in invalid_combinations:
        continue

    for year in years:
        # 动态生成每年的增长率和波动率
        growth_rate_sale_wheat_corn = np.random.uniform(0.05, 0.10)  # 小麦和玉米的销售增长率
        price_variation_other = np.random.uniform(-0.05, 0.05)  # 其他作物的价格变动范围
        yield_variation = np.random.uniform(-0.10, 0.10)  # 亩产量变动范围
        price_decline_mushroom = np.random.uniform(-0.05, -0.01)  # 食用菌价格下滑

        # 调整销售价格
        if crop_name in ['小麦', '玉米']:
            sale_price = avg_sale_price * (1 + growth_rate_sale_wheat_corn) ** (year - 2023)
        elif crop_name == '羊肚菌':
            sale_price = avg_sale_price * (1 + price_decline_yangdujun) ** (year - 2023)
        elif crop_name == '食用菌':
            sale_price = avg_sale_price * (1 + price_decline_mushroom) ** (year - 2023)
        else:
            sale_price = avg_sale_price * (1 + price_variation_other)

        # 调整销售价格范围和平均值
        min_price = min_sale_price * (1 + yield_variation)
        max_price = max_sale_price * (1 + yield_variation)
        avg_price = (min_price + max_price) / 2

        # 调整亩产量和成本
        yield_adj = yield_per_acre * (1 + yield_variation)
        cost_adj = plant_cost * (1 + cost_growth) ** (year - 2023)
        total_yield = yield_adj * plant_area

        # 存储模拟数据
        simulated_data['序号'].append(idx + 1)
        simulated_data['作物编号'].append(crop_num)
        simulated_data['作物名称'].append(crop_name)
        simulated_data['地块类型'].append(land_type)
        simulated_data['种植季次'].append(season_type)
        simulated_data['亩产量/斤'].append(yield_adj)
        simulated_data['种植成本/(元/亩)'].append(cost_adj)
        simulated_data['销售单价/(元/斤)'].append(f"{min_price:.2f}-{max_price:.2f}")
        simulated_data['最小销售单价'].append(min_price)
        simulated_data['最大销售单价'].append(max_price)
        simulated_data['平均销售单价'].append(avg_price)
        simulated_data['种植面积/亩'].append(plant_area)
        simulated_data['生产量/斤'].append(total_yield)
        simulated_data['年份'].append(year)  # 添加年份数据

# 将模拟数据转换为 DataFrame
simulated_df = pd.DataFrame(simulated_data)

# 清洗和标准化数据
data_cleaned = simulated_df.copy()
numeric_cols = data_cleaned.select_dtypes(include=[np.number])
data_cleaned[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())

# 导出清洗后的数据到 Excel
output_file_path = "simulated_data_output9.xlsx"
data_cleaned.to_excel(output_file_path, index=False, sheet_name="Simulated Data")

print(f"模拟数据已成功导出至 {output_file_path}")
