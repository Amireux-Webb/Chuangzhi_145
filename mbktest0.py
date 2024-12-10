import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint_sequential

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint_sequential

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 检查 CUDA 可用性
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 将模型包装为并行模型
    model = model.cuda()  # 移动到 GPU
else:
    print("CUDA is not available. Running on CPU.")

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器在 model.cuda() 之后定义

# 检查模型位置
print(f"Model is on: {next(model.parameters()).device}")

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        print(f"Inputs on: {inputs.device}, Labels on: {labels.device}")  # 验证数据位置
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
