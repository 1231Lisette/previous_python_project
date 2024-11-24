import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

# 读取 Excel 数据
att1 = pd.read_excel("附件1.xlsx", sheet_name=["乡村的现有耕地", "乡村种植的农作物"])
att2 = pd.read_excel("附件2.xlsx", sheet_name=["2023年的农作物种植情况", "2023年统计的相关数据"])
att3 = pd.read_excel("附件3.xlsx")


# 提取地块数据和作物数据
# 提取种植情况和统计数据
# 处理种植季次信息
# 为每种地块类型指定适宜种植季节
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

# 合并地块信息和作物信息
# 移除适宜种植季节的缺失值
# 使用合并后的正确列名，并处理数据类型和缺失值
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

crops_df = pd.merge(crops_basic_data, stats_data[['作物编号', '销售单价/(元/斤)', '种植成本/(元/亩)', '亩产量/斤']], on='作物编号', how='left')
crops_df = pd.DataFrame(crops)
crops_df = pd.merge(crops_df, att3[['作物名称', '预期销售量']], on='作物名称', how='left')

for crop in crops:
    for stat in stats_info:
        if crop['作物编号'] == stat['作物编号']:
            crop.update(stat)


plots = plots_data[['地块名称', '地块类型', '地块面积/亩']].to_dict('records')
plot_crop_df = plot_crop_df[plot_crop_df['适宜种植']]


model = LpProblem(name="Crop-Optimization", sense=LpMaximize)  # 定义线性规划模型，目标是最大化总利润
years = range(2024, 2031)
x = LpVariable.dicts("x", ((i['地块名称'], j['作物名称'], t) for i in plots_data.to_dict('records') for j in crops_df.to_dict('records') for t in years), lowBound=0, cat="Continuous")
z = LpVariable.dicts("z", ((i['地块名称'], j['作物名称'], t) for i in plots_data.to_dict('records') for j in crops_df.to_dict('records') for t in years), lowBound=0, cat="Continuous")
y = LpVariable.dicts("Crop_Planting",
                      [(i['地块名称'], j['作物名称'], t) for i in plots_data.to_dict('records') for j in crops_df.to_dict('records') for t in years],
                      cat='Binary')


model += lpSum(
    z[i['地块名称'], j['作物名称'], t] * j['销售单价/(元/斤)'] - x[i['地块名称'], j['作物名称'], t] * j['种植成本/(元/亩)']
    for i in plots_data.to_dict('records') for j in crops_df.to_dict('records') for t in years
)

# 添加约束条件
# 约束1：每个地块每年的种植面积不能超过地块面积
for i in plots_data.to_dict('records'):
    for t in years:
        model += lpSum(x[i['地块名称'], j['作物名称'], t] for j in crops_basic_data.to_dict('records')) <= i['地块面积/亩'], f"Total_Area_Constraint_{i['地块名称']}_{t}"
# 约束2：销售量不能超过预期销售量，且销售量不能超过种植量
for i in plots_data.to_dict('records'):
    for j in crops_df.to_dict('records'):
        for t in years:
            model += z[i['地块名称'], j['作物名称'], t] <= j.get('预期销售量', float('inf')), f"Sales_Limit_{i['地块名称']}_{j['作物名称']}_{t}"
            model += z[i['地块名称'], j['作物名称'], t] <= x[i['地块名称'], j['作物名称'], t] * j['亩产量/斤'], f"Production_Limit_{i['地块名称']}_{j['作物名称']}_{t}"
# 约束3：每个地块的同一作物在相邻年份不能重复种植
for i in plots_data.to_dict('records'):
    for j in crops_df.to_dict('records'):
        for t in years[:-1]:
            model += (
                y[i['地块名称'], j['作物名称'], t] + y[i['地块名称'], j['作物名称'], t + 1] <= 1,
                f"No_Replanting_{i['地块名称']}_{j['作物名称']}_{t}"
            )
for i in plots_data.to_dict('records'):  # 遍历每个地块
    # 约束 1：每个三年时间段内，至少种植一次豆类作物
    for t in range(2023, 2028 - 2):  # 从2023年开始，直到2025年（确保有足够的3年区间）
        model += lpSum(
            y[i['地块名称'], j['作物名称'], year]
            for j in crops_df.to_dict('records')
            if j['作物类型'] in ['蔬菜（豆类）', '粮食（豆类）']
            for year in range(t, t + 3)
        ) >= 1, f"Bean_Crop_Requirement_{i['地块名称']}_{t}"

    # 约束 2：如果种了豆类作物，则不能种其他作物
    for t in years:  # 遍历每个年份
        # 计算该地块在年份 t 是否种植了豆类作物
        bean_crop_sum = lpSum(
            y[i['地块名称'], j['作物名称'], t]
            for j in crops_basic_data.to_dict('records')
            if j['作物类型'] in ['蔬菜（豆类）', '粮食（豆类）']
        )
        # 计算该地块在年份 t 是否种植了非豆类作物
        non_bean_crop_sum = lpSum(
            y[i['地块名称'], j['作物名称'], t]
            for j in crops_basic_data.to_dict('records')
            if j['作物类型'] not in ['蔬菜（豆类）', '粮食（豆类）']
        )
        # 确保在同一年内只能种植豆类或其他作物中的一种
        model += bean_crop_sum + non_bean_crop_sum <= 1, f"No_Other_Crop_With_Beans_{i['地块名称']}_{t}"

# 约束5：只有适宜季节的作物才能种植
for i in plots_data.to_dict('records'):
    for j in crops_basic_data.to_dict('records'):
        for t in years:
            if not any(season in j['适宜种植季节'] for season in plots_data.loc[plots_data['地块名称'] == i['地块名称'], '适宜种植季节'].values[0]):
                model += y[i['地块名称'], j['作物名称'], t] == 0

# 约束6：平旱地、梯田和山坡地只能种植单季作物
for i in plots_data[plots_data['地块类型'].isin(['平旱地', '梯田', '山坡地'])].to_dict('records'):
    for t in years:
        model += lpSum(y[i['地块名称'], j['作物名称'], t] for j in crops_basic_data[crops_basic_data['适宜种植季节'].apply(lambda x: '单季' in x)].to_dict('records')) <= 1
# 约束7：水浇地的作物安排
for i in plots_data[plots_data['地块类型'] == '水浇地'].to_dict('records'):
    for t in years:
        model += lpSum(y[i['地块名称'], j['作物名称'], t] for j in crops_basic_data[crops_basic_data['适宜种植季节'].apply(lambda x: '第一季' in x or '第二季' in x)].to_dict('records')) <= 2
# 约束8：普通大棚种植安排
for i in plots_data[plots_data['地块类型'] == '普通大棚'].to_dict('records'):
    for t in years:
        model += lpSum(y[i['地块名称'], j['作物名称'], t] for j in crops_basic_data[crops_basic_data['适宜种植季节'].apply(lambda x: '第一季' in x)].to_dict('records')) <= 1
        model += lpSum(y[i['地块名称'], j['作物名称'], t] for j in crops_basic_data[crops_basic_data['适宜种植季节'].apply(lambda x: '第二季' in x)].to_dict('records')) <= 1
# 约束9：智慧大棚种植安排
for i in plots_data[plots_data['地块类型'] == '智慧大棚'].to_dict('records'):
    for t in years:
        model += lpSum(y[i['地块名称'], j['作物名称'], t] for j in crops_basic_data[crops_basic_data['适宜种植季节'].apply(lambda x: '第一季' in x)].to_dict('records')) <= 1
        model += lpSum(y[i['地块名称'], j['作物名称'], t] for j in crops_basic_data[crops_basic_data['适宜种植季节'].apply(lambda x: '第二季' in x)].to_dict('records')) <= 1

solver = PULP_CBC_CMD()
status = model.solve(solver)

if LpStatus[status] == 'Optimal':
    print("最优解找到！")
else:
    print("没有找到最优解！")


solution = pd.DataFrame(
    [(i, j, t, x[i, j, t].varValue) for i in plots_data['地块名称'] for j in crops_basic_data['作物名称'] for t in years if x[i, j, t].varValue > 0],
    columns=["地块名称", "作物名称", "年份", "种植面积"]
)


solution['利润'] = (
    solution.apply(lambda row: row['种植面积'] * crops_basic_data[crops_basic_data['作物名称'] == row['作物名称']]['销售单价/(元/斤)'].values[0] - row['种植面积'] * crops_basic_data[crops_basic_data['作物名称'] == row['作物名称']]['种植成本/(元/亩)'].values[0], axis=1)
)

annual_profit = solution.groupby(['年份'])['利润'].sum().reset_index()
total_profit = solution['利润'].sum()


with pd.ExcelWriter("优化结果.xlsx") as writer:
    solution.to_excel(writer, sheet_name="种植情况", index=False)
    annual_profit.to_excel(writer, sheet_name="每年利润", index=False)
    pd.DataFrame({'总利润': [total_profit]}).to_excel(writer, sheet_name="总利润", index=False)

