from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Step 1: 导入数据集
class TrafficSignDataset(Dataset):
    def __init__(self, folder_path, csv_file, transform=None, is_training=False):
        self.data = []
        self.transform = transform
        self.label_map = {}  # 创建一个标签映射字典

        # 读取图像数据
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) >= 2:  # 确保CSV文件至少有两列
                    label_str, label_char = row[0], row[1]
                    label = int(label_str)  # 将字符串标签转换为整数
                    self.label_map[label] = label_char  # 创建标签映射字典
                    label_folder = os.path.join(folder_path, label_str)
                    if os.path.isdir(label_folder):  # 检查标签文件夹是否存在
                        for filename in os.listdir(label_folder):
                            if filename.endswith('.png'):  # 仅处理图像文件
                                img_path = os.path.join(label_folder, filename)
                                img = Image.open(img_path)
                                if img is not None:
                                    img = img.convert("RGB")
                                    self.data.append((img, label))

        # 打印数据数量
        print(f"Total images in dataset: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.data = []
        self.transform = transform

        # 读取单个图像文件夹中的所有图像文件
        for filename in os.listdir(folder_path):
            img = Image.open(os.path.join(folder_path, filename))
            if img is not None:
                img = img.convert("RGB")  # 转换为RGB格式
                self.data.append((img, 0))  # 假设测试集中的标签为0

        # 打印数据数量
        print(f"Total images in dataset: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Step 2: 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Resize((46, 46)),  # 调整图像大小
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Step 3: 建立模型
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 减少输出通道数
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 减少输出通道数
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 减少输出通道数
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # 减少输出通道数
        self.fc1 = nn.Linear(256 * 2 * 2, 128)  # 减少全连接层节点数量
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.7)  # 增加 Dropout 的比例
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.batch_norm3(F.relu(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.batch_norm4(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Step 4: 模型训练
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0) / len(train_loader.dataset)
        print(f"  Epoch Loss: {running_loss:.4f}")

# Step 5: 模型评估
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)  # 只将图像张量移动到设备上
            labels = labels.to(device)  # 将标签张量也移动到设备上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        print("Test dataset is empty.")
    else:
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

# 设置路径和文件名
train_folder = "E:\\data\\data1"
csv_file = "E:\\data\\labels1.csv"
test_folder = "E:\\data\\test1"
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据集
train_dataset = TrafficSignDataset(train_folder, csv_file=csv_file, transform=transform)
test_dataset = TestDataset(test_folder, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型和优化器
model = CNN(num_classes=58).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
train_model(model, train_loader, criterion, optimizer, device)

# 评估模型
evaluate_model(model, test_loader, device)

# 使用模型进行预测
predictions = []
model.eval()
with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predicted_labels = [train_dataset.label_map[label.item()] for label in predicted]  # 将数值标签转换成名称标签
        predictions.extend(zip(filenames, predicted_labels))

# 将预测结果写入CSV文件
output_csv = "predictions1.csv"
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "predicted_label"])
    writer.writerows(predictions)

print("Predictions saved to", output_csv)
