import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
font = {'family': 'Microsoft YaHei', 'weight': 'bold'}
matplotlib.rc("font", **font)

# 读取 Excel 文件
df = pd.read_excel("D:\\桌面\\merged(1).xlsx")

# 确保“销售日期”列是 datetime 类型
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 定义函数来根据月份划分季节
def get_season(month):
    if month in [3, 4, 5]:
        return '春'
    elif month in [6, 7, 8]:
        return '夏'
    elif month in [9, 10, 11]:
        return '秋'
    elif month in [12, 1, 2]:
        return '冬'

# 添加“季节”列
df['季节'] = df['销售日期'].dt.month.apply(get_season)

# 提取年份和季节
df['年份'] = df['销售日期'].dt.year

# 按年份、季节和单品名称分组，并计算销售量总和
seasonal_sales = df.groupby(['单品名称', '年份', '季节'])['销量'].sum().unstack(level=[1, 2])

# 获取所有单品的总销售量
total_sales = seasonal_sales.sum(axis=1)

# 排序并选择前10名和后10名的单品
top_10 = total_sales.nlargest(10)
bottom_10 = total_sales.nsmallest(10)

# 选择前10名和后10名的单品数据
selected_top_data = seasonal_sales.loc[top_10.index]
selected_bottom_data = seasonal_sales.loc[bottom_10.index]

# 为不同季节定义颜色
season_colors = {
    '春': 'green',
    '夏': 'red',
    '秋': 'orange',
    '冬': 'blue'
}

# 绘制销量前10名的图
fig, ax = plt.subplots(figsize=(14, 8))  # 调整图的大小
width = 0.2  # 每个柱子的宽度
x = range(len(selected_top_data))  # x轴位置

# 存储图例标签
legend_labels = []

for i, (season, color) in enumerate(season_colors.items()):
    if season in selected_top_data.columns.get_level_values(1):
        season_data = selected_top_data.xs(season, level=1, axis=1)
        for j, year in enumerate(season_data.columns):
            year_data = season_data[year]
            bars = ax.bar(
                [p + width * i for p in x],  # 调整柱子的x轴位置
                year_data,  # 每个菜品在该季节和年份的总销售量
                width=width,  # 调整柱子的宽度
                color=color,
                label=f'{year} {season}',  # 年份和季节作为图例标签
                alpha=0.7
            )
            # 找出每个菜品在这个季节中的最高销售量并标注
            for k, bar in enumerate(bars):
                yval = bar.get_height()
                max_val = season_data.iloc[k].max()
                if yval == max_val:
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02*yval, round(yval, 1),
                            ha='center', va='bottom', fontsize=10)

            legend_labels.append(f'{year} {season}')

ax.set_title('销量前10名蔬菜的年度季节销售量分布')
ax.set_xlabel('菜品名称')
ax.set_ylabel('总销售量')
ax.set_xticks([p + width * (len(season_colors) / 2) for p in x])  # 调整x轴刻度的位置
ax.set_xticklabels(selected_top_data.index, rotation=45)  # 设置x轴刻度标签
plt.tight_layout()  # 调整布局以防止标签被遮挡
plt.legend(title='年份 季节', labels=legend_labels, loc='upper right')  # 添加图例
plt.show()

# 绘制销量后10名的图
fig, ax = plt.subplots(figsize=(14, 8))  # 调整图的大小
x = range(len(selected_bottom_data))  # x轴位置

# 存储图例标签
legend_labels = []

for i, (season, color) in enumerate(season_colors.items()):
    if season in selected_bottom_data.columns.get_level_values(1):
        season_data = selected_bottom_data.xs(season, level=1, axis=1)
        for j, year in enumerate(season_data.columns):
            year_data = season_data[year]
            bars = ax.bar(
                [p + width * i for p in x],  # 调整柱子的x轴位置
                year_data,  # 每个菜品在该季节和年份的总销售量
                width=width,  # 调整柱子的宽度
                color=color,
                label=f'{year} {season}',  # 年份和季节作为图例标签
                alpha=0.7
            )
            # 对于销量后10名的所有柱子都标注销售量值
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01*yval, round(yval, 1),
                        ha='center', va='bottom', fontsize=10)

            legend_labels.append(f'{year} {season}')

ax.set_title('销量后10名蔬菜的年度季节销售量分布')
ax.set_xlabel('菜品名称')
ax.set_ylabel('总销售量')
ax.set_xticks([p + width * (len(season_colors) / 2) for p in x])  # 调整x轴刻度的位置
ax.set_xticklabels(selected_bottom_data.index, rotation=45)  # 设置x轴刻度标签
plt.tight_layout()  # 调整布局以防止标签被遮挡
plt.legend(title='年份 季节', labels=legend_labels)  # 添加图例
plt.show()
