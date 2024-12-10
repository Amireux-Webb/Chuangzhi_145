import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.utils.checkpoint as checkpoint
from model import ModelArgs, Transformer
import argparse
import os

# Argument parser for distributed training
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank provided by torchrun")
    return parser.parse_args()

# Setup for distributed training
def setup_distributed(local_rank):
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

# Custom Transformer block with Activation Checkpointing
class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        def custom_forward(*inputs):
            h = inputs[0] + self.attention(self.attention_norm(inputs[0]), start_pos, freqs_cis, mask)
            return h + self.feed_forward(self.ffn_norm(h))

        return checkpoint.checkpoint(custom_forward, x)

# Trainer class for DDP
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")
        setup_distributed(args.local_rank)

        # Initialize model with DDP
        model_args = ModelArgs(dim=4096, n_layers=32, n_heads=32, max_seq_len=args.seq_len)
        model = Transformer(model_args)
        model.layers = nn.ModuleList([CheckpointedTransformerBlock(i, model_args) for i in range(model_args.n_layers)])
        self.model = DDP(model.to(self.device), device_ids=[args.local_rank], output_device=args.local_rank)

        # Data loader with DistributedSampler
        dataset = YourDataset()  # Replace with actual dataset
        self.sampler = DistributedSampler(dataset)
        self.dataloader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=self.sampler, num_workers=4
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

    def train_epoch(self):
        self.model.train()
        self.sampler.set_epoch(0)  # Shuffle data for each epoch
        for batch in self.dataloader:
            inputs, targets = batch["input_ids"].to(self.device), batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, start_pos=0)

            # Compute loss
            logits_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=2)(logits_flat, targets_flat)

            # Backward pass
            loss.backward()
            self.optimizer.step()

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch()
            if dist.get_rank() == 0:
                print(f"Epoch {epoch + 1} completed")

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
