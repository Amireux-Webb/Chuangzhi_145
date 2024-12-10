import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed_mode():
    dist.init_process_group(backend='nccl')  # 初始化分布式进程组
    torch.cuda.set_device(dist.get_rank())   # 设置当前进程使用的设备

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # 一个简单的全连接层

    def forward(self, x):
        return self.fc(x)

def train(rank, world_size):
    init_distributed_mode()  # 初始化分布式环境
    model = SimpleModel().cuda()  # 将模型转移到当前 GPU

    # 使用 DDP 包装模型
    model = DDP(model, device_ids=[rank])

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 示例数据（实际使用中可以用 DataLoader）
    data = torch.randn(100, 10).cuda()  # 100个样本，10个特征
    target = torch.randn(100, 1).cuda()  # 100个目标值

    for epoch in range(10):  # 训练 10 个 epoch
        model.train()

        # 前向传播
        output = model(data)
        loss = nn.MSELoss()(output, target)  # 计算均方误差损失

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新参数

        print(f'Rank {rank}, Epoch [{epoch+1}/10], Loss: {loss.item()}')

if __name__ == "__main__":
    world_size = 2  # 总的进程数（使用 2 个 GPU）
    os.environ['WORLD_SIZE'] = str(world_size)  # 设置环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '29500'  # 设置主节点端口

    processes = []
    for rank in range(world_size):
        os.environ['RANK'] = str(rank)  # 设置当前进程的 RANK
        os.environ['LOCAL_RANK'] = str(rank)  # 设置本地 GPU 的索引
        
        p = mp.Process(target=train, args=(rank, world_size))  # 启动进程
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # 等待所有进程完成
