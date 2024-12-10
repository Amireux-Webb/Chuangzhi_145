import torch
import torch.nn as nn
from torch.optim import AdamW
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
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint/", help= "checkpoint path")
    parser.add_argument("--data_dir", type=str, default="dataset/Tangshi/", help="dataset path")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoint/tokenizer.model", help="tokenizer path")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    
    return parser.parse_args()


class LLMTrainer:
    def __init__(self, args):
        
        torch.manual_seed(42)
        self.args = args
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
        self.model = Transformer(self.model_args)
        self.model.load_state_dict(self.checkpoint, strict=False)
        
        self.model.to(self.device)
        
        self.dataloader = get_PoemDataloader(data_dir=args.data_dir, batch_size=args.batch_size)
        self.optimizer = AdamW(self.model.parameters(), args.lr)
        
    # 定义模型训练函数
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Training", ncols=100, miniters=10):
            input_ids = batch["input_ids"].to(self.device)
            target = batch["labels"].to(self.device)
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(tokens=input_ids, start_pos=0)
            # 计算交叉熵损失
            logits_flat = outputs.view(-1, 32000)  # [B*S, V]
            targets_flat = target.view(-1)           # [B*S]

            # 使用 ignore_index 忽略 padding 部分
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=2)
            # 反向传播
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            
            
        return total_loss / len(self.dataloader)
    
    def train(self):
        # 训练循环   
        print("Training Start...", flush=True)
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
            if epoch%20 == 0:
                self.save_checkpoint(epoch)
                print(f"Saved checkpoint for epoch {epoch}, Training loss: {train_loss}")
                
        print("Training Finished!", flush=True)
        self.save_checkpoint(epoch)
        
            
    def save_checkpoint(self, epoch_idx):
        torch.save(self.model.state_dict(), f"./finetuned_llm_{epoch_idx}.pth")
        
        

    
if __name__ == "__main__":      
    args = parse_args()
    trainer = LLMTrainer(args)
    trainer.train()
    
    
    
    




    