import laspy
import numpy as np
from collections import Counter

# 读取 LAS 文件
file_path = "E:\\pcl\\data\\MANIFOLD_2024-6-16-Cloud_LAS.las"
las = laspy.read(file_path)

# 提取 RGB 颜色通道
red = las.red
green = las.green
blue = las.blue

# 找到 RGB 颜色通道中的最大值，用于确定缩放因子
max_value = max(max(red), max(green), max(blue))

# 将 RGB 颜色值缩放到 0-255 范围
scaled_red = np.clip((red * 255.0 / max_value).astype(int), 0, 255)
scaled_green = np.clip((green * 255.0 / max_value).astype(int), 0, 255)
scaled_blue = np.clip((blue * 255.0 / max_value).astype(int), 0, 255)

# 标识红色点（R=255, G=0, B=0）
red_indices = (scaled_red == 255) & (scaled_green == 0) & (scaled_blue == 0)

# 保留不是红色的点
filtered_indices = ~red_indices

# 创建新的 LAS 文件，保留过滤后的点
filtered_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
filtered_las.points = las.points[filtered_indices]

# 保存新的 LAS 文件
output_file_path = "E:\\pcl\\data\\filtered_cloud3.las"
filtered_las.write(output_file_path)

print(f"Filtered LAS file saved to {output_file_path}")
