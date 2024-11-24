import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

# 读取 Excel 数据
att1 = pd.read_excel("附件1.xlsx", sheet_name=["乡村的现有耕地", "乡村种植的农作物"])
att2 = pd.read_excel("附件2.xlsx", sheet_name=["2023年的农作物种植情况", "2023年统计的相关数据"])
att3 = pd.read_excel("附件3.xlsx")

# 提取地块数据和作物数据
plots_data = att1["乡村的现有耕地"]
crops_basic_data = att1["乡村种植的农作物"]

# 提取种植情况和统计数据
planting_data = att2["2023年的农作物种植情况"]
stats_data = att2["2023年统计的相关数据"]


# 处理种植季次信息
planting_data['种植季次'] = planting_data['种植季次'].map({'单季':  0, '第一季': 1, '第二季': 2})


# 为每种地块类型指定适宜种植季节
plots_data['适宜种植季节'] = plots_data['地块类型'].map({
    '平旱地': ['单季'],
    '梯田': ['单季'],
    '山坡地': ['单季'],
    '水浇地': ['单季', '第一季', '第二季'],
    '普通大棚': ['第一季', '第二季'],
    '智慧大棚': ['第一季', '第二季']
})

# 为每种作物指定适宜的种植季节
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
    '菠菜': ['第一季', '第二季'],  # 去除额外空格
    '青椒': ['第一季', '第二季'],
    '菜花': ['第一季', '第二季'],
    '包菜': ['第一季', '第二季'],
    '油麦菜': ['第一季', '第二季'],
    '小青菜': ['第一季', '第二季'],
    '黄瓜': ['第一季', '第二季'],
    '生菜': ['第一季', '第二季'],  # 去除额外空格
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
print("plots_data columns:", plots_data.columns.tolist())
print("crops_basic_data columns:", crops_basic_data.columns.tolist())


# 合并地块信息和作物信息
plot_crop_df = pd.merge(plots_data, crops_basic_data, how='cross')

# 移除适宜种植季节的缺失值
plot_crop_df.dropna(subset=['适宜种植季节_x', '适宜种植季节_y'], inplace=True)

# 使用合并后的正确列名，并处理数据类型和缺失值
plot_crop_df['适宜种植'] = plot_crop_df.apply(
    lambda row: any(str(season) in str(row['适宜种植季节_y']) for season in str(row['适宜种植季节_x']).split(',')),
    axis=1
)

# 筛选出适宜种植的记录
plot_crop_df = plot_crop_df[plot_crop_df['适宜种植']]

print(plot_crop_df.info())


# 处理销售单价数据，将价格区间转换为均值
def parse_price_range(price_range):
    if isinstance(price_range, str):
        try:
            low, high = price_range.split('-')
            return (float(low) + float(high)) / 2  # 返回区间的平均值
        except ValueError:
            return None
    return None

# 处理销售单价列，将价格区间转换为均值
if '销售单价/(元/斤)' in stats_data.columns:
    stats_data['销售单价/(元/斤)'] = stats_data['销售单价/(元/斤)'].apply(parse_price_range)

# 从基础作物数据中提取基本信息，并与销售数据结合
crops = crops_basic_data[['作物编号', '作物名称', '作物类型']].to_dict('records')
stats_info = stats_data[['作物编号', '销售单价/(元/斤)', '种植成本/(元/亩)', '亩产量/斤']].to_dict('records')

# 将销售数据与基础作物数据结合
for crop in crops:
    for stat in stats_info:
        if crop['作物编号'] == stat['作物编号']:
            crop.update(stat)

# 转换为 DataFrame，并合并预期销售量数据
crops_df = pd.DataFrame(crops)
crops_df = pd.merge(crops_df, att3[['作物名称', '预期销售量']], on='作物名称', how='left')

# 地块数据
plots = plots_data[['地块名称', '地块类型', '地块面积/亩']].to_dict('records')

# 创建优化模型
model = LpProblem(name="Crop-Optimization", sense=LpMaximize)  # 定义线性规划模型，目标是最大化总利润

# 定义年份范围
years = range(2024, 2031)  # 计划从 2024 年到 2030 年

# 定义决策变量
# x: 地块、作物、年份的连续型变量，表示种植面积
x = LpVariable.dicts("x", ((i['地块名称'], j['作物名称'], t) for i in plots for j in crops_df.to_dict('records') for t in years), lowBound=0, cat="Continuous")
# z: 地块、作物、年份的连续型变量，表示销售量
z = LpVariable.dicts("z", ((i['地块名称'], j['作物名称'], t) for i in plots for j in crops_df.to_dict('records') for t in years), lowBound=0, cat="Continuous")
# y: 地块、作物、年份的二元变量，表示是否种植
y = LpVariable.dicts("Crop_Planting",
                          [(i['地块名称'], j['作物名称'], t) for i in plots for j in crops_df.to_dict('records') for t in years],
                          cat='Binary')

# 定义目标函数：最大化总利润
model += lpSum(
    z[i['地块名称'], j['作物名称'], t] * j['销售单价/(元/斤)'] - x[i['地块名称'], j['作物名称'], t] * j['种植成本/(元/亩)']
    for i in plots for j in crops_df.to_dict('records') for t in years
)

# 添加约束条件

# 约束1：每个地块每年的种植面积不能超过地块面积
for i in plots:
    for t in years:
        model += lpSum(x[i['地块名称'], j['作物名称'], t] for j in crops_df.to_dict('records')) <= i['地块面积/亩'], f"Total_Area_Constraint_{i['地块名称']}_{t}"

# 约束2：销售量不能超过预期销售量，且销售量不能超过种植量
for i in plots:
    for j in crops_df.to_dict('records'):
        for t in years:
            model += z[i['地块名称'], j['作物名称'], t] <= j['预期销售量'], f"Sales_Limit_{i['地块名称']}_{j['作物名称']}_{t}"
            model += z[i['地块名称'], j['作物名称'], t] <= x[i['地块名称'], j['作物名称'], t] * j['亩产量/斤'], f"Production_Limit_{i['地块名称']}_{j['作物名称']}_{t}"

# 约束3：每个地块的同一作物在相邻年份不能重复种植
for i in plots:
    for j in crops_df.to_dict('records'):
        for t in years[:-1]:
            model += (
                y[i['地块名称'], j['作物名称'], t] + y[i['地块名称'], j['作物名称'], t + 1] <= 1,
                f"No_Replanting_{i['地块名称']}_{j['作物名称']}_{t}"
            )

# 约束4：每块地在某些年份必须种植豆类作物
for i in plots:
    for t in range(2023, 2028 - 2):  # 从2023年开始，直到2025年（确保有足够的3年区间）
        model += lpSum(
            y[i['地块名称'], j['作物名称'], year]
            for j in crops_df.to_dict('records')
            if j['作物类型'] in ['蔬菜（豆类）', '粮食（豆类）']
            for year in range(t, t + 3)
        ) >= 1, f"Legume_Constraint_{i['地块名称']}_{t}"

# 约束5：二元变量约束，确保种植面积不超过地块面积
for i in plots:
    for j in crops_df.to_dict('records'):
        for t in years:
            model += x[i['地块名称'], j['作物名称'], t] <= y[i['地块名称'], j['作物名称'], t] * i['地块面积/亩'], f"Binary_Constraint_{i['地块名称']}_{j['作物名称']}_{t}"

# 约束6：地块类型与作物类型匹配约束
for i in plots:
    for j in crops_df.to_dict('records'):
        for t in years:
            if j['作物名称'] not in plots_data.loc[plots_data['地块名称'] == i['地块名称'], '适宜种植季节'].values:
                model += y[i['地块名称'], j['作物名称'], t] == 0, f"Type_Mismatch_{i['地块名称']}_{j['作物名称']}_{t}"

# 求解模型
solver = PULP_CBC_CMD(msg=True)
model.solve(solver)

# 打印结果
print(f"状态: {LpStatus[model.status]}")
print(f"总利润: {model.objective.value()}")


# 创建一个列表来保存结果
results = []

# 打印每块地块、每种作物、每年种植面积、销售量
for i in plots:
    for j in crops_df.to_dict('records'):
        for t in years:
            if x[i['地块名称'], j['作物名称'], t].varValue > 0:
                print(f"地块: {i['地块名称']}, 作物: {j['作物名称']}, 年份: {t}")
                print(f"  种植面积: {x[i['地块名称'], j['作物名称'], t].varValue} 亩")
                print(f"  销售量: {z[i['地块名称'], j['作物名称'], t].varValue} 斤")


