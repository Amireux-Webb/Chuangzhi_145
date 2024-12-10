import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from dataloader import get_PoemDataloader
import json
from mbkmodel import ModelArgs, Transformer
from tokenizer import Tokenizer
from pathlib import Path
import torch.nn.functional as F
import tqdm 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint/", help="checkpoint path")
    parser.add_argument("--data_dir", type=str, default="dataset/Tangshi/", help="dataset path")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoint/tokenizer.model", help="tokenizer path")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=16, help="sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank provided by torchrun")
    return parser.parse_args()

# Initialize distributed training environment
def setup_distributed(local_rank):
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

class LLMTrainer:
    def __init__(self, args):
        torch.manual_seed(42)
        self.args = args

        # 使用 local_rank 初始化 device
        self.device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

        # 初始化分布式环境
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)

        with open(Path(args.ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        self.model_args: ModelArgs = ModelArgs(
            max_seq_len=args.seq_len,
            max_batch_size=args.batch_size,
            **params,
        )
        self.tokenizer = Tokenizer(model_path=args.tokenizer_path)
        self.model_args.vocab_size = self.tokenizer.n_words

        print("Loading checkpoint...")
        self.checkpoint = torch.load(args.ckpt_dir + "llm.pth", map_location="cpu", weights_only=True)
        print("Loading checkpoint done!")
        model = Transformer(self.model_args)
        model.load_state_dict(self.checkpoint, strict=False)

        # 封装模型为 DDP
        self.model = DDP(model.to(self.device), device_ids=[args.local_rank], output_device=args.local_rank)

        # 使用 DistributedSampler 确保多卡分布式数据加载
        dataset = get_PoemDataloader(data_dir=args.data_dir, batch_size=args.batch_size)
        self.sampler = DistributedSampler(dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=self.sampler
        )

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)


    def train_epoch(self):
        self.model.train()
        self.sampler.set_epoch(0)  # Shuffle data for each epoch
        total_loss = 0

        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.dataloader), total=len(self.dataloader), desc="Training", ncols=100, miniters=10
        ):
            input_ids = batch["input_ids"].to(self.device)
            target = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(tokens=input_ids, start_pos=0)

            # Compute cross-entropy loss
            logits_flat = outputs.view(-1, self.model_args.vocab_size)
            targets_flat = target.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=2)

            # Backward and optimization
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.dataloader)

    def train(self):
        print("Training Start...", flush=True)
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            if dist.get_rank() == 0:  # Only the main process logs
                print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

            if dist.get_rank() == 0 and epoch % 20 == 0:
                self.save_checkpoint(epoch)
                print(f"Saved checkpoint for epoch {epoch}, Training loss: {train_loss}")

        if dist.get_rank() == 0:
            print("Training Finished!", flush=True)
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch_idx):
        if dist.get_rank() == 0:  # Only save on main process
            torch.save(self.model.module.state_dict(), f"./finetuned_llm_{epoch_idx}.pth")

if __name__ == "__main__":
    args = parse_args()
    trainer = LLMTrainer(args)
    trainer.train()
