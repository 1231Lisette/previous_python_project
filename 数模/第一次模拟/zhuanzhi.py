import pandas as pd

# 读取Excel文件
input_file = "D:\\桌面\\test2.xlsx"  # 输入文件路径
output_file = "D:\\桌面\\output_transposed.xlsx"  # 输出文件路径

# 读取Excel文件中的第一个工作表
df = pd.read_excel(input_file, sheet_name=0)

# 转置数据
df_transposed = df.T

# 保存转置后的数据到新的Excel文件
df_transposed.to_excel(output_file, index=False)

print(f"转置后的数据已保存到 {output_file}")
