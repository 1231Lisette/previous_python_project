import numpy as np
# 导入numpy库，用于进行科学计算
# 从用户输入中接收参评数目和指标数目，并将输入的字符串转换为数值

print("请输入参评数目:")
n = int(input())    # 接收参评数目
print("请输入指标数目:")
m = int(input())  # 接收指标数目

# 接收用户输入的类型矩阵，该矩阵指示了每个指标的类型(极大型、极小型等)
print("请输入类型矩阵:1:极大型，2:极小型，3:中间型，4:区间型")
kind = input().split("")    # 将输入的字符串按空格分割，形成列表

# 接收用户输入的矩阵并转换为numpy数组
print("请输入矩阵:")
A = np.zeros(shape=(n, m))   # 初始化一个n行m列的全零矩阵A
for i in range(n):
    A[i] = input().split("")    # 接收每行输入的数据
    A[i] = list(map(float, A[i]))       # 将接收到的字符串列表转换为浮点数列表
print("输入矩阵为:\n{}", format(A))       # 打印输入的矩阵A

def min_to_max(max_x, x):
    x = list(x)
    ans = [[max_x - e] for e in x]
    return np.array(ans)

def mid_to_max(best_x, x):
    x=list(x)   # 将输入的指标数据转换为列表
    h = [abs(e-best_x)for e in x]   # 计算每个指标值与最优值之间的绝对差
    M = max(h)  # 找到最大的差值
    if M == 0:
        M = 1   # 防止最大差值为0的情况
    ans = [[(1-e/M)]for e in h]     # 计算每个差值占最大差值的比例，并从1中减去，得到新指标值
    return np.array(ans)    # 返回处理后的numpy数组

def reg_to_max(low_x, high_x,x):
    x = list(x)  # 将输入的指标数据转换为列表#计算指标值超出区间的最大距离
    M = max(low_x-min(x),max(x)-high_x)
    if M == 0:
        M = 1  # 防止最大距离为0的情况
    ans = []
    for i in range(len(x)):
        if x[i] < low_x:
            ans.append([(1 - (low_x - x[i]) / M)])  # 如果指标值小于下限，则计算其与下限的距离比例
        elif x[i] > high_x:
            ans.append([(1 - (x[i] - high_x) / M)])  # 如果指标值大于上限，则计算其与上限的距离比例
        else:
            ans.append([1])  # 如果指标值在区间内，则直接取为1
    return np.array(ans)

# 统一指标类型，将所有指标转化为极大型指标
X = np.zeros(shape = (n, 1))
for i in range(m):
    if kind[i] == "1":
        v = np.array(A[:, i])
    elif kind[i] == "2":
        maxA = max(A[:, i])
        v = min_to_max(maxA, A[:, i])
    elif kind[i] == "3":
        print("类型3，请输入最优值：")
        bestA = eval(input())
        v = mid_to_max(bestA,A[:, i])
    elif kind[i] == "4":
        print("类型4，请输入区间[a, b]的a")
        lowA = eval(input())
        print("类型4，请输入区间[a, b]的b")
        highA = eval(input())
        v = reg_to_max(lowA, highA,A[:, i])
    if i == 0:
        X = v.reshape(-1, 1)
    else:
        X = np.hstack([X, v.reshape(-1, 1)])
print("统一指标后的矩阵为：\n{}".format(X))

# 对统一指标后的矩阵X进行标准化处理
X = X.astype('float') # 确保X矩阵的数据类型为浮点数
for j in range(m):
    X[:, j]= X[:, j]/np.sqrt(sum(X[:, j]**2)) # 对每一列数据进行归一化处理，即除以该列的欧几里得范数
print("标准化矩阵为:\n{}".format(X))  # 打印标准化后的矩阵X

# 最大值最小值距离的计算
x_max = np.max(X, axis=0)   # 计算标准化矩阵每列的最大值
x_min = np.min(X, axis=0)   # 计算标准化矩阵每列的最小值
d_z = np.sqrt(np.sum(np.square((X - np.tile(x_max, (n, 1)))), axis=1))
d_f = np.sqrt(np.sum(np.square((X - np.tile(x_min, (n, 1)))), axis=1))
print("每个指标的最大值：", x_max)
print("每个指标的最小值：", x_min)
print('d+向量:', d_z)
print('d-向量:', d_f)

# 计算每个参评对象的得分排名
s = d_f/(d_z+d_f)    # 根据d+和d-计算得分s，其中s接近于1则表示较优，接近于0则表示较劣
Score=100*s/sum(s)  # 将得分s转换为百分制，便于比较
for i in range(len(Score)):
    print(f"第{i+1}个标准化后百分制得分为:{Score[i]}")  # 打印每个参评对象的得分


