import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib

# 中文字体配置
font = {'family': 'Microsoft YaHei', 'weight': 'bold'}
matplotlib.rc("font", **font)
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# Step 1: 从 Excel 文件中导入数据
file_path = "D:\\桌面\\数模\\C题\\att2_sale_production.xlsx"  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 显示数据的前几行以确认加载正确
print(data.head())

# Step 2: 生成未来几年的模拟数据（2024-2030）
years = list(range(2024, 2031))

# 定义参数
growth_rate_sale_wheat_corn = np.random.uniform(0.05, 0.10)  # 小麦和玉米的销售量增长率（5%-10%）
price_variation_other_crops = np.random.uniform(-0.05, 0.05)  # 其他作物的价格变动范围（-5%到+5%）
yield_variation = np.random.uniform(-0.10, 0.10)  # 亩产量变动范围（-10%到+10%）
cost_growth = 0.05  # 成本增长率（5%每年）
price_decline_mushroom = np.random.uniform(-0.05, -0.01)  # 食用菌价格下滑（-5%到-1%）
price_decline_yangdujun = -0.05  # 羊肚菌价格下滑（-5%）
price_growth_vegetable = 0.05  # 蔬菜价格年增长（5%）

# 创建存储模拟数据的字典
simulated_data = {
    'year': [],
    'crop_name': [],
    'plot_type': [],
    'predicted_sale_price': [],
    'predicted_yield': [],
    'predicted_cost': [],
    'predicted_sales_volume': []
}

# Step 3: 遍历每种作物和年份生成模拟数据
for _, row in data.iterrows():
    crop_name = row['crop_name']
    avg_sale_price = row['average_sale_price']
    yield_per_acre = row['yield']
    plant_cost = row['plant_money']
    plot_type = row['land_type']  # 地块类型（例如：平旱地、梯田、山坡地、水浇地、普通大棚、智慧大棚）

    for year in years:
        # 调整销售量
        if crop_name in ['小麦', '玉米']:  # 小麦和玉米的销售量增长趋势
            predicted_sales_volume = yield_per_acre * (1 + growth_rate_sale_wheat_corn) ** (year - 2023)
        else:  # 其他作物的销售量随机变化在±5%范围内
            predicted_sales_volume = yield_per_acre * (1 + price_variation_other_crops)

        # 调整销售价格
        if crop_name in ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '谷子', '高粱', '黍子', '荞麦', '南瓜', '红薯', '莜麦', '大麦', '水稻']:  # 粮食类价格稳定
            sale_price = avg_sale_price
        elif crop_name in ['羊肚菌']:  # 羊肚菌价格下降5%
            sale_price = avg_sale_price * (1 + price_decline_yangdujun) ** (year - 2023)
        elif crop_name in ['榆黄菇', '香菇', '白灵菇']:  # 其他食用菌价格下降1%-5%
            sale_price = avg_sale_price * (1 + price_decline_mushroom) ** (year - 2023)
        elif crop_name in ['豇豆', '刀豆', '芸豆', '土豆', '西红柿', '茄子', '菠菜', '青椒', '菜花', '包菜', '油麦菜', '小青菜', '黄瓜', '生菜', '辣椒', '空心菜', '黄心菜', '芹菜', '大白菜', '白萝卜', '红萝卜']:  # 蔬菜类价格上涨5%
            sale_price = avg_sale_price * (1 + price_growth_vegetable) ** (year - 2023)

        # 调整亩产量和成本
        if plot_type in ['普通大棚', '智慧大棚']:  # 双季作物
            yield_adj = yield_per_acre * (1 + yield_variation)
            cost_adj = plant_cost * (1 + cost_growth) ** (year - 2023)
        else:  # 单季作物
            yield_adj = yield_per_acre * (1 + yield_variation)
            cost_adj = plant_cost * (1 + cost_growth) ** (year - 2023)

        # 存储模拟数据
        simulated_data['year'].append(year)
        simulated_data['crop_name'].append(crop_name)
        simulated_data['plot_type'].append(plot_type)
        simulated_data['predicted_sale_price'].append(sale_price)
        simulated_data['predicted_yield'].append(yield_adj)
        simulated_data['predicted_cost'].append(cost_adj)
        simulated_data['predicted_sales_volume'].append(predicted_sales_volume)

# 将模拟数据转换为 DataFrame
simulated_df = pd.DataFrame(simulated_data)

# Step 4: 清洗和标准化数据
data_cleaned = simulated_df.copy()

# 填充缺失值，只选择数值列
numeric_cols = data_cleaned.select_dtypes(include=[np.number])
data_cleaned[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())

# 检查并去除异常值（使用 z-score 方法）
z_scores = np.abs(stats.zscore(data_cleaned[['predicted_sale_price', 'predicted_yield', 'predicted_cost', 'predicted_sales_volume']]))
data_cleaned = data_cleaned[(z_scores < 3).all(axis=1)]

# 确保数据类型正确（例如，'year' 应为整数）
data_cleaned['year'] = data_cleaned['year'].astype(int)

# Step 5: 数据可视化
plt.figure(figsize=(14, 8))
sns.lineplot(data=data_cleaned, x='year', y='predicted_sale_price', hue='crop_name', marker='o')
plt.title('预测销售价格随年份变化')
plt.xlabel('年份')
plt.ylabel('预测销售价格')
plt.legend(title='作物名称', fontsize='small', loc='upper left', bbox_to_anchor=(1., 1.05))
plt.show()

plt.figure(figsize=(14, 8))
sns.lineplot(data=data_cleaned, x='year', y='predicted_yield', hue='crop_name', marker='o')
plt.title('预测亩产量随年份变化')
plt.xlabel('年份')
plt.ylabel('预测亩产量')
plt.legend(title='作物名称', fontsize='small', loc='upper left', bbox_to_anchor=(1., 1.05))
plt.show()

plt.figure(figsize=(14, 8))
sns.lineplot(data=data_cleaned, x='year', y='predicted_cost', hue='crop_name', marker='o')
plt.title('预测种植成本随年份变化')
plt.xlabel('年份')
plt.ylabel('预测种植成本')
plt.legend(title='作物名称', fontsize='small', loc='upper left', bbox_to_anchor=(1, 1.05))
plt.show()

plt.figure(figsize=(14, 8))
sns.lineplot(data=data_cleaned, x='year', y='predicted_sales_volume', hue='crop_name', marker='o')
plt.title('预测销售量随年份变化')
plt.xlabel('年份')
plt.ylabel('预测销售量')
plt.legend(title='作物名称', fontsize='small', loc='upper left', bbox_to_anchor=(1, 1.05))
plt.show()

# Step 6: 导出数据
data_cleaned.to_excel('cleaned_data1.xlsx', index=False)
