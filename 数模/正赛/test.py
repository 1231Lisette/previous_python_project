import pandas as pd

# 读取Excel文件中的两个sheet
file_path = "D:\\桌面\\数模\\C题\\附件2.xlsx"  # 请替换为你的实际文件路径
sheet1 = pd.read_excel(file_path, sheet_name='2023年的农作物种植情况')
sheet2 = pd.read_excel(file_path, sheet_name='2023年统计的相关数据')

# 根据地块编号的开头字母，定义一个函数来确定地块类型
# 根据地块编号的开头字母，定义一个函数来确定地块类型
def determine_land_type(land_code):
    if str(land_code).startswith('A'):
        return '平旱地'
    elif str(land_code).startswith('B'):
        return '梯田'
    elif str(land_code).startswith('C'):
        return '山坡地'
    elif str(land_code).startswith('D'):
        return '水浇地'
    elif str(land_code).startswith('E'):
        return '普通大棚'
    elif str(land_code).startswith('F'):
        return '智慧大棚'
    else:
        return '未知'

# 在 sheet1 中添加一个地块类型列
sheet1['地块类型'] = sheet1['种植地块'].apply(determine_land_type)

# 根据作物编号和地块类型进行合并
merged_data = pd.merge(sheet1, sheet2, how='left', left_on=['作物编号', '地块类型'], right_on=['作物编号', '地块类型'])

# 保存合并后的数据到新的Excel文件
output_file = "D:\\桌面\\数模\\C题\\整合后的数据.xlsx"
merged_data.to_excel(output_file, index=False)

print(f"整合完成，结果已保存为 '{output_file}'")
