import torch
import torch.nn as nn
from torch.optim import AdamW
from dataloader import get_PoemDataloader
import json
from mbkmodel_ac2 import ModelArgs, Transformer
from tokenizer import Tokenizer
from pathlib import Path
import torch.nn.functional as F
import tqdm
import argparse
import os
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/checkpoint", help="checkpoint path")  # 修改路径
    parser.add_argument("--data_dir", type=str, default="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/dataset/Tangshi", help="dataset path")
    parser.add_argument("--tokenizer_path", type=str, default="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/checkpoint/tokenizer.model", help="tokenizer path")  # 修改路径
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    
    return parser.parse_args()


class LLMTrainer:
    def __init__(self, args, rank, world_size):
        torch.manual_seed(42 + rank)  # 二次传入rank确保可复现性
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        with open(Path(args.ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        self.model_args: ModelArgs = ModelArgs(
            max_seq_len=args.seq_len,
            max_batch_size=args.batch_size,
            **params,
        )
        self.tokenizer = Tokenizer(model_path=args.tokenizer_path)
        self.model_args.vocab_size = self.tokenizer.n_words
        print(f"Rank {rank}: Loading checkpoint...")
        self.checkpoint = torch.load(args.ckpt_dir + "/llm.pth", map_location="cpu", weights_only=True)  # 修改路径
        print(f"Rank {rank}: Loading checkpoint done!")

        self.model = Transformer(self.model_args)
        self.model.load_state_dict(self.checkpoint, strict=False)
        self.model.to(self.device)

        # 使用 DistributedSampler
        self.dataloader = get_PoemDataloader(data_dir=args.data_dir, batch_size=args.batch_size)
        self.optimizer = AdamW(self.model.parameters(), args.lr)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Training", ncols=100, miniters=10):
            input_ids = batch["input_ids"].to(self.device)
            target = batch["labels"].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(tokens=input_ids, start_pos=0)
            logits_flat = outputs.view(-1, 32000)
            targets_flat = target.view(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=2)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.dataloader)

    def train(self):
        print(f"Rank {self.rank}: Training Start...", flush=True)
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            if self.rank == 0:  # 只在rank 0打印
                print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
            if epoch % 20 == 0 and self.rank == 0:
                self.save_checkpoint(epoch)
                print(f"Rank {self.rank}: Saved checkpoint for epoch {epoch}, Training loss: {train_loss}")

        print(f"Rank {self.rank}: Training Finished!", flush=True)
        if self.rank == 0:
            self.save_checkpoint(self.args.epochs)

    def save_checkpoint(self, epoch_idx):
        torch.save(self.model.state_dict(), f"./finetuned_llm_{epoch_idx}.pth")


def main(rank, world_size):
    args = parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'  # 或者主节点的真实地址
    os.environ['MASTER_PORT'] = '12345'  # 确保该端口未被其他进程占用

    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    trainer = LLMTrainer(args, rank, world_size)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 获取可用 GPU 数量
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)  # 启动多个进程进行训练
