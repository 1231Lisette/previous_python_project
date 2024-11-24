import numpy as np

A = np.array([[1, 2, 3, 5], [1/2, 1, 1/2, 2], [1/3, 2, 1, 2,], [1/5, 1/2, 1/2, 1]])
n = A.shape[0]  # 0为行，1为列

eig_val, eig_vec = np.linalg.eig(A)
Max_eig = max(eig_val)  # 求特征值的最大值

CI = (Max_eig - n) / (n-1)
RI = [0, 0.0001, 0.52, 0.89, 1.12,  1.26,  1.36, 1.41, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]

CR = CI / RI[n-1]
print('一致性指标CI=', CI)
print('一致性比例CR=', CR)

if CR < 0.10:
    print('判断矩阵的一致性可以接受。')
else:
    print('判断矩阵的一致性不理想，需要重新调整。')

# 找出最大特征值的索引
max_index = np.argmax(eig_val)

# 找出对应的特征向量
max_vector = eig_vec[:, max_index]

# 对特征向量进行归一化处理，得到权重
weight = max_vector / np.sum(max_vector)

# 输出权重
print(weight)