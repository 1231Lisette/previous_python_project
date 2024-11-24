import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# 读取 Excel 数据
att1 = pd.read_excel("附件1.xlsx")
att2 = pd.read_excel("附件2.xlsx")


# 将数据转换为字典列表
plots = att1[['地块名称', '地块类型', '地块面积/亩']].to_dict('records')
crops = att2[['作物名称', '作物类型', '种植面积/亩']].to_dict('records')

# 创建优化模型
model = LpProblem(name="Crop-Optimization", sense=LpMaximize)

# 定义年份范围
years = range(2024, 2031)

# 定义决策变量
x = LpVariable.dicts("x", ((i['地块名称'], j['作物名称'], t) for i in plots for j in crops for t in years), lowBound=0, cat="Continuous")
y = LpVariable.dicts("y", ((i['地块名称'], j['作物名称'], t) for i in plots for j in crops for t in years), lowBound=0, upBound=1, cat="Binary")

# 定义目标函数
model += lpSum(x[i['地块名称'], j['作物名称'], t] * j['种植面积/亩'] for i in plots for j in crops for t in years), "Total Planting Area"

# 约束条件 1：每个地块每年的总种植面积不能超过地块面积
for i in plots:
    for t in years:
        model += lpSum(x[i['地块名称'], j['作物名称'], t] for j in crops) <= i['地块面积/亩']

# 约束条件 2：不同作物可同时种植，但不能超过地块面积
for i in plots:
    for j in crops:
        for t in years:
            model += x[i['地块名称'], j['作物名称'], t] <= y[i['地块名称'], j['作物名称'], t] * i['地块面积/亩']

# 约束条件 3：相邻两年不种植相同作物
for i in plots:
    for j in crops:
        for t in range(2025, 2031):
            model += y[i['地块名称'], j['作物名称'], t] + y[i['地块名称'], j['作物名称'], t - 1] <= 1

# 约束条件 4：每个地块在三年内至少种植一次豆类作物
for i in plots:
    for t in range(2024, 2028):
        model += lpSum(y[i['地块名称'], j['作物名称'], t + k] for j in crops if j['作物类型'] in ['蔬菜（豆类）', '粮食（豆类）'] for k in range(3)) >= 1

# 求解模型
status = model.solve()

# 输出求解状态和结果
print(f"求解状态：{model.status}")
for i in plots:
    for j in crops:
        for t in years:
            if x[i['地块名称'], j['作物名称'], t].value() > 0:
                print(f"地块{i['地块名称']} 在第 {t} 年种植 {j['作物名称']} {x[i['地块名称'], j['作物名称'], t].value()} 亩")

# 创建一个列表来保存结果
results = []

# 收集求解结果
for i in plots:
    for j in crops:
        for t in years:
            planted_area = x[i['地块名称'], j['作物名称'], t].value()
            if planted_area > 0:
                results.append({
                    '地块名称': i['地块名称'],
                    '作物名称': j['作物名称'],
                    '年份': t,
                    '种植面积/亩': planted_area
                })

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results)

# 导出为 Excel 文件
results_df.to_excel('种植优化结果2.xlsx', index=False)

print("结果已导出到 '种植优化结果2.xlsx'")
