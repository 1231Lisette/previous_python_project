import torch
import numpy as np
import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(c)
print("Vectorized version:" + str(1000*(toc-tic)) + "ms")

c1 = 0
tic1 = time.time()
for i in range(1000000):
    c1 += a[i]*b[i]
toc1 = time.time()
print(c1)
print("For loop:" + str(1000*(toc1-tic1)) + "ms")
# t = torch.cuda.is_available()
# print(t)

