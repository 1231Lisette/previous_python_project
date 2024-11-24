import pandas as pd
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, GLPK, LpStatus
import numpy as np

att1 = pd.read_excel("附件1.xlsx", sheet_name=["乡村的现有耕地", "乡村种植的农作物"])
att2 = pd.read_excel("附件2.xlsx", sheet_name=["2023年的农作物种植情况", "2023年统计的相关数据"])
att3 = pd.read_excel("附件3.xlsx")

plots_data = att1["乡村的现有耕地"]
crops_basic_data = att1["乡村种植的农作物"]


planting_data = att2["2023年的农作物种植情况"]
stats_data = att2["2023年统计的相关数据"]
planting_data['种植季次'] = planting_data['种植季次'].map({'单季': 0, '第一季': 1, '第二季': 2})
plots_data['适宜种植季节'] = plots_data['地块类型'].map({
    '平旱地': ['单季'],
    '梯田': ['单季'],
    '山坡地': ['单季'],
    '水浇地': ['单季', '第一季', '第二季'],
    '普通大棚': ['第一季', '第二季'],
    '智慧大棚': ['第一季', '第二季']
})
crops_basic_data['适宜种植季节'] = crops_basic_data['作物名称'].map({
    '黄豆': ['单季'],
    '黑豆': ['单季'],
    '红豆': ['单季'],
    '绿豆': ['单季'],
    '爬豆': ['单季'],
    '小麦': ['单季'],
    '玉米': ['单季'],
    '谷子': ['单季'],
    '高粱': ['单季'],
    '黍子': ['单季'],
    '荞麦': ['单季'],
    '南瓜': ['单季'],
    '红薯': ['单季'],
    '莜麦': ['单季'],
    '大麦': ['单季'],
    '水稻': ['单季'],
    '豇豆': ['第一季', '第二季'],
    '刀豆': ['第一季', '第二季'],
    '芸豆': ['第一季', '第二季'],
    '土豆': ['第一季', '第二季'],
    '西红柿': ['第一季', '第二季'],
    '茄子': ['第一季', '第二季'],
    '菠菜': ['第一季', '第二季'],
    '青椒': ['第一季', '第二季'],
    '菜花': ['第一季', '第二季'],
    '包菜': ['第一季', '第二季'],
    '油麦菜': ['第一季', '第二季'],
    '小青菜': ['第一季', '第二季'],
    '黄瓜': ['第一季', '第二季'],
    '生菜': ['第一季', '第二季'],
    '辣椒': ['第一季', '第二季'],
    '空心菜': ['第一季', '第二季'],
    '黄心菜': ['第一季', '第二季'],
    '芹菜': ['第一季', '第二季'],
    '大白菜': ['第二季'],
    '白萝卜': ['第二季'],
    '红萝卜': ['第二季'],
    '榆黄菇': ['第二季'],
    '香菇': ['第二季'],
    '白灵菇': ['第二季'],
    '羊肚菌': ['第二季']
})

plot_crop_df = pd.merge(plots_data, crops_basic_data, how='cross')
plot_crop_df.dropna(subset=['适宜种植季节_x', '适宜种植季节_y'], inplace=True)
plot_crop_df['适宜种植'] = plot_crop_df.apply(
    lambda row: any(season in row['适宜种植季节_y'] for season in row['适宜种植季节_x']),
    axis=1
)
def parse_price_range(price_range):
    if isinstance(price_range, str):
        try:
            low, high = price_range.split('-')
            return (float(low) + float(high)) / 2  # 返回区间的平均值
        except ValueError:
            return None
    return None

if '销售单价/(元/斤)' in stats_data.columns:
    stats_data['销售单价/(元/斤)'] = stats_data['销售单价/(元/斤)'].apply(parse_price_range)
crops = crops_basic_data[['作物编号', '作物名称', '作物类型']].to_dict('records')
stats_info = stats_data[['作物编号', '销售单价/(元/斤)', '种植成本/(元/亩)', '亩产量/斤']].to_dict('records')

for crop in crops:
    for stat in stats_info:
        if crop['作物编号'] == stat['作物编号']:
            crop.update(stat)

crops_df = pd.DataFrame(crops)
crops_df = pd.merge(crops_df, att3[['作物名称', '预期销售量']], on='作物名称', how='left')

plots = plots_data[['地块名称', '地块类型', '地块面积/亩']].to_dict('records')
plot_crop_df = plot_crop_df[plot_crop_df['适宜种植']]
years = range(2024, 2031)
n_simulations = 50  # 设置模拟次数
all_results = []  # 用于存储每次模拟的结果

def adjust_value(value, growth_rate, years):
    return [value * ((1 + growth_rate) ** (year - years[0])) for year in years]

for sim in range(n_simulations):
    # 更新作物数据
    for idx, crop in crops_df.iterrows():
        crop_name = crop['作物名称']
        if crop_name in ['小麦', '玉米']:
            sales_volume_growth_rate = np.random.uniform(0.05, 0.10)
        else:
            sales_volume_growth_rate = np.random.uniform(-0.05, 0.05)

        yield_variability = np.random.uniform(-0.10, 0.10)
        cost_increase_rate = 0.05

        if crop['作物类型'] == '粮食':
            price_growth_rate = 0
        elif crop['作物类型'] == '蔬菜':
            price_growth_rate = 0.05
        elif crop['作物类型'] == '食用菌':
            price_growth_rate = np.random.uniform(-0.05, -0.01)

        # 更新作物的预期销售量、亩产量、种植成本和销售价格
        crop['预期销售量'] = adjust_value(crop['预期销售量'], sales_volume_growth_rate, years)
        crop['亩产量/斤'] = adjust_value(crop['亩产量/斤'], yield_variability, years)
        crop['种植成本/(元/亩)'] = adjust_value(crop['种植成本/(元/亩)'], cost_increase_rate, years)
        crop['销售单价/(元/斤)'] = adjust_value(crop['销售单价/(元/斤)'], price_growth_rate, years)

    # 创建优化模型
    model = LpProblem(name=f"Crop-Optimization-Simulation-{sim}", sense=LpMaximize)

    # 定义决策变量
    x = {}
    for plot in plots:
        for crop in crops_df.itertuples():
            x[(plot['地块名称'], crop.作物编号)] = LpVariable(
                f"x_{plot['地块名称']}_{crop.作物编号}",
                lowBound=0,
                upBound=plot['地块面积/亩'],
                cat='Continuous'
            )

    # 定义目标函数
    model += lpSum(
        x[(plot['地块名称'], crop.作物编号)] * crop.销售单价/(元/斤) * crop.亩产量/斤
        for plot in plots
        for crop in crops_df.itertuples()
        if (plot['地块名称'], crop.作物编号) in x
    ), "Total_Profit"

    # 定义约束条件
    for plot in plots:
        model += lpSum(
            x[(plot['地块名称'], crop.作物编号)]
            for crop in crops_df.itertuples()
            if (plot['地块名称'], crop.作物编号) in x
        ) <= plot['地块面积/亩'], f"Area_Constraint_{plot['地块名称']}"

    for crop in crops_df.itertuples():
        model += lpSum(
            x[(plot['地块名称'], crop.作物编号)]
            for plot in plots
            if (plot['地块名称'], crop.作物编号) in x
        ) <= crop.预期销售量[-1], f"Sales_Volume_Constraint_{crop.作物编号}"

    # 求解模型
    solver = pulp.GLPK()
    model.solve(solver)

    # 记录结果
    result = {
        'Simulation': sim,
        'Status': LpStatus[model.status],
        'Objective': model.objective.value(),
        'Variable_Values': {var.name: var.varValue for var in model.variables()}
    }
    all_results.append(result)

# 输出模拟结果
results_df = pd.DataFrame(all_results)

