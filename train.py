import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from dataloader import get_PoemDataloader
import json
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from pathlib import Path
import torch.nn.functional as F
import tqdm 
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint/", help= "checkpoint path")
    parser.add_argument("--data_dir", type=str, default="dataset/Tangshi/", help="dataset path")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoint/tokenizer.model", help="tokenizer path")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    
    return parser.parse_args()


class LLMTrainer:
    def __init__(self, args):
        dist.init_process_group(backend="nccl", init_method="env://")# 并行计算
                
        torch.manual_seed(42)
        self.args = args
        self.rank = dist.get_rank()
        # print(f"{self.rank=}")
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        self.full_model = Transformer(self.model_args)
        self.full_model.load_state_dict(self.checkpoint, strict=False)
        self.model_part = self.split_model(self.full_model)
        
        self.model_part.to(self.device)
        
        self.dataloader = get_PoemDataloader(data_dir=args.data_dir, batch_size=args.batch_size)
        self.optimizer = AdamW(self.model_part.parameters(), args.lr)
        self.criterion = nn.CrossEntropyLoss()
       
    # 按照模型不同层进行分割 
    def split_model(self, full_model):
        num_layers = len(full_model.layers)
        layers_per_stage = num_layers // self.world_size
        print(f"The model has {num_layers} layers and {layers_per_stage} for each GPU")
        print(f"{self.rank=}")
        # 存疑：nn.Sequential的结构体
        if self.rank == 0:
            submodule = nn.Sequential(
                full_model.tok_embeddings,
                *full_model.layers[:layers_per_stage],
            )
        elif self.rank == self.world_size - 1:
            submodule = nn.Sequential(
                *full_model.layers[self.rank * layers_per_stage:],
                full_model.norm,
                full_model.output,
            )
        else:
            submodule = nn.Sequential(
                *full_model.layers[self.rank * layers_per_stage: (self.rank + 1) * layers_per_stage],
            )
        return submodule

    # 定义前向传播函数
    def forward_pipeline(self, inputs):
        if self.rank == 0:
            inputs = inputs.to(self.device)
            outputs = self.model_part(inputs)
            dist.send(tensor=outputs, dst=self.rank + 1)
            return None
        elif self.rank == self.world_size - 1:
            recv_inputs = torch.empty_like(inputs, device=self.device)
            dist.recv(tensor=recv_inputs, src=self.rank - 1)
            outputs = self.model_part(recv_inputs)
            return outputs
        else:
            recv_inputs = torch.empty_like(inputs, device=self.device)
            dist.recv(tensor=recv_inputs, src=self.rank - 1)
            outputs = self.model_part(recv_inputs)
            dist.send(tensor=outputs, dst=self.rank + 1)
            return None
    
    # 定义反向传播函数
    def backward_pipeline(self, grad_outputs):
        if self.rank == self.world_size - 1:
            grad_outputs = grad_outputs.to(self.device)
            grad_outputs.backward()
            dist.send(tensor=grad_outputs.grad, dst=self.rank - 1)
        elif self.rank == 0:
            recv_grad = torch.empty_like(grad_outputs, device=self.device)
            dist.recv(tensor=recv_grad, src=self.rank + 1)
            recv_grad.backward()
        else:
            recv_grad = torch.empty_like(grad_outputs, device=self.device)
            dist.recv(tensor=recv_grad, src=self.rank + 1)
            recv_grad.backward()
            dist.send(tensor=recv_grad.grad, dst=self.rank - 1)

    # 定义模型训练函数       
    def train_epoch(self):
        """
        单个 epoch 的训练逻辑，支持分布式流水线。
        """
        self.model_part.train()  # 确保模型在训练模式
        total_loss = 0

        # 遍历数据加载器，支持 tqdm 显示进度条
        for inputs, targets in tqdm(self.dataloader, desc=f"[Rank {self.rank}] Training Progress"):
            if self.rank == 0:
                # 第 0 阶段：移动目标到设备，并发送到最后一个阶段
                targets = targets.to(self.device)
                dist.send(tensor=targets, dst=self.world_size - 1)

            # 前向传播流水线
            outputs = self.forward_pipeline(inputs)

            if self.rank == self.world_size - 1:
                # 最后一个阶段：计算损失并开始反向传播
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                dist.send(tensor=loss.grad, dst=self.rank - 1)
                total_loss += loss.item()
            else:
                # 中间阶段：反向传播流水线
                self.backward_pipeline(outputs)

            # 每个阶段都需要更新自己的参数
            self.optimizer.step()

        return total_loss / len(self.dataloader)

            
    def train(self):
        """
        分布式流水线训练主函数
        """
        print("Training Start...", flush=True)
        for epoch in range(self.args.epochs):
            # 调用模块化的训练逻辑
            train_loss = self.train_epoch()
            print(f"[Rank {self.rank}] Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

            # 每隔 20 个 epoch 保存一次检查点
            if epoch % 20 == 0:
                self.save_checkpoint(epoch)
                print(f"[Rank {self.rank}] Saved checkpoint at epoch {epoch + 1} with loss {train_loss:.4f}")

        print(f"[Rank {self.rank}] Training Finished!", flush=True)
        self.save_checkpoint(epoch)    
            
    def save_checkpoint(self, epoch):
        """
        保存模型检查点的功能。
        """
        checkpoint = {
            "model_state_dict": self.model_part.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "rank": self.rank,
        }
        checkpoint_path = f"./finetuned_llm_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        # print(f"[Rank {self.rank}] Checkpoint saved to {checkpoint_path}", flush=True)
        
if __name__ == "__main__":      
    args = parse_args()
    trainer = LLMTrainer(args)
    trainer.train()