import torch
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv("D:\\桌面\\田字型散点.csv", header=None)
data = pd.DataFrame(data)

# 提取特征和标签
X = torch.tensor(data.iloc[0:2, :].values, dtype=torch.float32)
labels = torch.tensor(data.iloc[2, :].values, dtype=torch.float32)

# 定义神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(2, 4)  # 输入特征维度为2，隐藏层维度为64
        self.linear2 = nn.Linear(4, 1)   # 隐藏层维度为64，输出维度为1
        self.relu = nn.ReLU()             # 使用ReLU作为激活函数

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))

# 实例化模型
model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，使用较小的学习率

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.T, labels, test_size=0.2, random_state=42)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))  # view(-1, 1)将标签形状调整为列向量

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练过程中的损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    outputs = model(X_test)
    predicted_labels = torch.round(outputs)
    accuracy = (predicted_labels == y_test.view(-1, 1)).sum().item() / y_test.size(0)
    print(f'Test Accuracy: {accuracy:.2f}')
