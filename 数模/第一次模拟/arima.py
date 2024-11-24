import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 1. 数据筛选
# 加载数据
merged_df = pd.read_excel("D:\\桌面\\merged.xlsx", sheet_name='Sheet1')

# 转换日期格式
merged_df['date'] = pd.to_datetime(merged_df['date'])

# 筛选出夏季数据（6月、7月、8月）
summer_data = merged_df[(merged_df['date'].dt.month.isin([6, 7, 8])) &
                        (merged_df['date'].dt.year.isin([2020, 2021, 2022, 2023]))]

# 保存筛选后的数据
summer_data.to_excel('summer_data_filtered.xlsx', index=False)

# 2. 分析各蔬菜品类的销售总量与成本加成定价的关系
# 加载批发价数据
wholesale_price_df = pd.read_excel('attachment_4.xlsx', sheet_name='Sheet1')

# 将夏季销售数据与批发价数据合并
merged_sales_price_df = pd.merge(summer_data, wholesale_price_df, on=['single_name', 'date'], how='left')

# 计算各蔬菜品类的销售总量与批发价格的关系
X = merged_sales_price_df[['wholesale_price']]
y = merged_sales_price_df['sales_volume']

# 添加常数项用于回归分析
X = sm.add_constant(X)

# 回归模型
model = sm.OLS(y, X).fit()
summary = model.summary()

# 打印回归结果
print(summary)

# 使用回归系数计算新的销售预测
coefficient = model.params['wholesale_price']
intercept = model.params['const']
merged_sales_price_df['predicted_sales_volume'] = coefficient * merged_sales_price_df['wholesale_price'] + intercept

# 3. 利用 ARIMA 模型进行未来一周的销售量预测
replenishment_forecasts = []

for category, group in merged_sales_price_df.groupby('category'):
    # 按时间顺序排序
    group = group.sort_values('date')

    # 使用 ARIMA 模型进行时间序列预测
    model = ARIMA(group['sales_volume'], order=(5, 1, 0))  # order 可调，根据 ACF/PACF 图决定
    model_fit = model.fit()

    # 预测未来 7 天的销售量
    forecast = model_fit.forecast(steps=7)

    # 保存预测结果
    replenishment_forecasts.append({
        'category': category,
        'predicted_sales': forecast.sum()
    })

# 将预测结果转换为 DataFrame
forecast_df = pd.DataFrame(replenishment_forecasts)

# 4. 结合预测数据、批发价格和损耗率计算补货量和定价策略
final_strategy = pd.merge(forecast_df, merged_sales_price_df[['category', 'wholesale_price', 'loss_rate']],
                          on='category', how='left')
final_strategy['adjusted_replenishment'] = final_strategy['predicted_sales'] * (1 + final_strategy['loss_rate'])
final_strategy['final_price'] = final_strategy['wholesale_price'] * (1 + 0.2)  # 假设加成率为 20%

# 5. 计算收益
final_strategy['revenue'] = final_strategy['final_price'] * final_strategy['predicted_sales'] - final_strategy[
    'wholesale_price'] * final_strategy['adjusted_replenishment']
total_revenue = final_strategy.groupby('category')['revenue'].sum().reset_index()

# 保存收益计算结果
final_strategy.to_excel('final_replenishment_pricing_revenue_strategy.xlsx', index=False)
total_revenue.to_excel('total_revenue_by_category.xlsx', index=False)

# 6. 可视化图表生成

# 收益柱状图
plt.figure(figsize=(10, 6))
plt.bar(total_revenue['category'], total_revenue['revenue'], color='green')
plt.xlabel('蔬菜品类')
plt.ylabel('总收益')
plt.title('各蔬菜品类的总收益')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_revenue_by_category.png')
plt.show()

# 定价策略与收益的关系图
plt.figure(figsize=(10, 6))
plt.scatter(final_strategy['final_price'], final_strategy['revenue'], color='blue')
plt.xlabel('定价策略 (元/公斤)')
plt.ylabel('收益 (元)')
plt.title('定价策略与收益的关系')
plt.grid(True)
plt.savefig('pricing_vs_revenue.png')
plt.show()

# 打印各品类总收益
print("各品类总收益:\n", total_revenue)
