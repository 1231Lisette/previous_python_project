import torch
import torchvision
from torchvision import transforms

# 从pytorch下载fashion-MINIST数据集🤔


# 将图像数据转换为tensor类型并对其标准化
# 第一个元组(0.5,)表示数据的均值，第二个元组(0.5,)表示数据的标准差。
transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))))

# 下载训练集
train_dataset = torchvision.datasets.FashionMNIST(root='./root', train=True, download=True)

# 训练集数据加载器
batch_size = 64     # 指定批次大小
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# shuffle=True表示每次迭代时打乱数据

# 下载测试集
test_dataset = torchvision.datasets.FashionMNIST(root='./root', train=False, download=True)

# 测试集数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 查看特征数量
# Fashion-MNIST数据集中的特征是图像数据，因此可以通过shape属性查看图像的大小
print("训练集特征数量:", train_dataset.data.shape)
print("测试集特征数量:", test_dataset.data.shape)
# 查看标签数量
# Fashion-MNIST数据集中的标签是类别数字，可以通过classes属性查看类别的数量
print("训练集标签数量:", len(train_dataset.classes))
print("测试集标签数量:", len(test_dataset.classes))

# 训练集60000个样本，每个样本是一个大小为28x28的二维张量（图像）。每个图像都有28行和28列。
# 测试集10000个样本

