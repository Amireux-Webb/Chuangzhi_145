import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from dataloader import get_PoemDataloader
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from pathlib import Path
import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/checkpoint/", help="checkpoint path")
    parser.add_argument("--data_dir", type=str, default="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/dataset/Tangshi/", help="dataset path")  # 更新为绝对路径
    parser.add_argument("--tokenizer_path", type=str, default="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/checkpoint/tokenizer.model", help="tokenizer path")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    return parser.parse_args()

# 其他代码保持不变


def init_distributed_mode():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

class LLMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用绝对路径打开 params.json 文件
        params_path = Path(args.ckpt_dir) / "params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"Cannot find the params.json file at {params_path}")
        
        with open(params_path, "r") as f:
            params = json.load(f)

        self.model_args: ModelArgs = ModelArgs(
            max_seq_len=args.seq_len,
            max_batch_size=args.batch_size,
            **params,
        )
        
        # 使用绝对路径加载 tokenizer.model
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Cannot find the tokenizer model file at {tokenizer_path}")
        
        self.tokenizer = Tokenizer(model_path=str(tokenizer_path))
        self.model_args.vocab_size = self.tokenizer.n_words       
        print("Loading checkpoint...")
        checkpoint_path = Path(args.ckpt_dir) / "llm.pth"
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print("Loading checkpoint done!")
        
        self.model = Transformer(self.model_args)
        self.model.load_state_dict(self.checkpoint, strict=False)
        self.model.to(self.device)
        
        # 设置数据加载器
        self.dataloader = get_PoemDataloader(data_dir=str(Path(args.data_dir)), batch_size=args.batch_size)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Training", ncols=100, miniters=10):
            input_ids = batch["input_ids"].to(self.device)
            target = batch["labels"].to(self.device)

            # Forward pass
            output = self.model(tokens=input_ids, start_pos=0)
            logits_flat = output.view(-1, self.model_args.vocab_size)
            targets_flat = target.view(-1)

            # 计算交叉熵损失
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=2)
            total_loss += loss.item()

            # 梯度清零
            self.model.zero_grad()

            # 反向传播
            loss.backward()
            
            # 手动更新参数并进行梯度同步
            for param in self.model.parameters():
                dist.all_reduce(param.grad.data)  # 所有 GPU 上的梯度累加
                param.data -= self.args.lr * (param.grad.data / dist.get_world_size())  # 手动更新参数

        return total_loss / len(self.dataloader)

    def train(self):
        print("Training Start...", flush=True)
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
            if epoch % 20 == 0:
                self.save_checkpoint(epoch)
                print(f"Saved checkpoint for epoch {epoch}, Training loss: {train_loss}")

        print("Training Finished!", flush=True)
        self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch_idx):
        to_save = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(to_save, str(Path("finetuned_llm_{}.pth".format(epoch_idx))))

def spawn_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 本地地址
    os.environ['MASTER_PORT'] = '12345'  # 有效的端口号
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    init_distributed_mode()
    args = parse_args()

    trainer = LLMTrainer(args)
    trainer.train()

if __name__ == "__main__":
    world_size = 2  # 设置为您要使用的 GPU 数量
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(target=spawn_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # 等待所有进程完成
