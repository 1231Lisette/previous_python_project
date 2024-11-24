def coinchange(n):
    """
    用于计算找委的最少硬币数。
    参数n:要找零的金额返回值
    最少硬币数量，如果无法找零，则返回-1
    """
    dp = [float('inf')] * (n+1)     # 初始化动态数组
    dp[0] = 0   # 找零金额为0时，需要0枚硬币
    for i in range(1, n+1):
        if i >= 2:
            dp[i] = min(dp[i], dp[i-2]+1)
        if i >= 5:
            dp[i] = min(dp[i], dp[i-5]+1)
        if i >= 7:
            dp[i] = min(dp[i], dp[i-7]+1)
    if dp[n] != float('inf'):
        return dp[n]
    else:
        return -1

n = int(input('请输入你要拼的金额：'))
res = coinchange(n)
print(res)