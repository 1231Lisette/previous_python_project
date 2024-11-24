import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义类别名称
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# 加载测试数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = torchvision.datasets.FashionMNIST(root='data/FashionMNIST', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载训练好的模型
PATH = 'my_model.pth'
net = Net()  # 假设Net是你之前定义的模型
net.load_state_dict(torch.load(PATH))  # 加载模型参数
net.eval()  # 设置模型为评估模式

# 获取一批测试图像和标签
for images, labels in test_loader:
    # 预测图像类别
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # 可视化预测结果
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {classes[predicted[i]]}\nActual: {classes[labels[i]]}')
    plt.show()
