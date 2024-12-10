import os
import json
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
import torch


class PoemDataset(Dataset):
    def __init__(self, data_dir, max_length=16):
        self.data_dir = data_dir
        self.tokenizer = Tokenizer(model_path="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/checkpoint/tokenizer.model")
        self.max_length = max_length
        self.data_pairs = self._prepare_data()

    def _prepare_data(self):
        """
        从 JSON 文件中提取诗句，并构造上下句对
        :return: (input, target) 对的列表
        """
        data_pairs = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(self.data_dir, file_name), "r", encoding="utf-8") as f:
                    poems = json.load(f)  # 假设 JSON 格式为标准的中国诗词格式
                    for poem in poems:
                        paragraphs = poem.get("paragraphs", [])
                        for paragraph in paragraphs:
                            # 按逗号分隔上下句
                            sentences = paragraph.split("，")
                            # 确保至少有两句，跳过无效段落
                            if len(sentences) < 2:
                                continue
                            for i in range(len(sentences) - 1):
                                input_text = sentences[i].strip() + "，"  # 上句
                                target_text = sentences[i + 1].strip()   # 下句
                                # 确保上下句非空
                                if input_text and target_text:
                                    data_pairs.append((input_text, target_text))
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    
    def __getitem__(self, idx):
        input_text, target_text = self.data_pairs[idx]

        # 分词
        input_ids = self.tokenizer.encode(input_text, bos=False, eos=False)
        labels = self.tokenizer.encode(target_text, bos=False, eos=True)

        # 转换为 1D Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # 填充到固定长度
        input_ids = self._pad_sequence(input_ids, bsz=1).squeeze(0)
        labels = self._pad_sequence(labels, bsz=1).squeeze(0)


        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def _pad_sequence(self, seq, bsz=1):
        """
        填充序列到最大长度（支持批量处理）
        :param seq: 输入的 token 序列（1D Tensor 或 2D Tensor）
        :param bsz: 批量大小，默认为 1
        :return: 填充后的序列（形状 [bsz, max_length]）
        """
        pad_id = self.tokenizer.eos_id
        total_len = self.max_length

        # 初始化全填充的张量
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)

        # 如果 seq 是 1D Tensor，将其直接复制到 tokens 中
        if seq.ndimension() == 1:
            length = min(len(seq), total_len)
            tokens[0, :length] = seq[:length]
        elif seq.ndimension() == 2:  # 如果是 2D Tensor，支持批量填充
            for i in range(min(bsz, seq.size(0))):
                length = min(len(seq[i]), total_len)
                tokens[i, :length] = seq[i][:length]
        else:
            raise ValueError("Input seq must be 1D or 2D Tensor.")

        return tokens

# 创建 DataLoader
def get_PoemDataloader(data_dir, batch_size=32, shuffle=True):
    """
    获取 DataLoader
    :param data_dir: 数据文件夹路径
    :param batch_size: 批量大小
    :param shuffle: 是否打乱数据
    :param num_workers: 数据加载的线程数
    :return: PyTorch DataLoader
    """
    dataset = PoemDataset(data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 使用示例
if __name__ == "__main__":
    data_dir = "/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/dataset/Tangshi"  # 替换为实际路径
    dataloader = get_PoemDataloader(data_dir, batch_size=4)
    for batch in dataloader:
        input_data = batch["input_ids"]
        target = batch["labels"]
        
        break
import json
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
import torch


class PoemDataset(Dataset):
    def __init__(self, data_dir, max_length=16):
        self.data_dir = data_dir
        self.tokenizer = Tokenizer(model_path="/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/checkpoint/tokenizer.model")
        self.max_length = max_length
        self.data_pairs = self._prepare_data()

    def _prepare_data(self):
        """
        从 JSON 文件中提取诗句，并构造上下句对
        :return: (input, target) 对的列表
        """
        data_pairs = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(self.data_dir, file_name), "r", encoding="utf-8") as f:
                    poems = json.load(f)  # 假设 JSON 格式为标准的中国诗词格式
                    for poem in poems:
                        paragraphs = poem.get("paragraphs", [])
                        for paragraph in paragraphs:
                            # 按逗号分隔上下句
                            sentences = paragraph.split("，")
                            # 确保至少有两句，跳过无效段落
                            if len(sentences) < 2:
                                continue
                            for i in range(len(sentences) - 1):
                                input_text = sentences[i].strip() + "，"  # 上句
                                target_text = sentences[i + 1].strip()   # 下句
                                # 确保上下句非空
                                if input_text and target_text:
                                    data_pairs.append((input_text, target_text))
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    
    def __getitem__(self, idx):
        input_text, target_text = self.data_pairs[idx]

        # 分词
        input_ids = self.tokenizer.encode(input_text, bos=False, eos=False)
        labels = self.tokenizer.encode(target_text, bos=False, eos=True)

        # 转换为 1D Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # 填充到固定长度
        input_ids = self._pad_sequence(input_ids, bsz=1).squeeze(0)
        labels = self._pad_sequence(labels, bsz=1).squeeze(0)


        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def _pad_sequence(self, seq, bsz=1):
        """
        填充序列到最大长度（支持批量处理）
        :param seq: 输入的 token 序列（1D Tensor 或 2D Tensor）
        :param bsz: 批量大小，默认为 1
        :return: 填充后的序列（形状 [bsz, max_length]）
        """
        pad_id = self.tokenizer.eos_id
        total_len = self.max_length

        # 初始化全填充的张量
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)

        # 如果 seq 是 1D Tensor，将其直接复制到 tokens 中
        if seq.ndimension() == 1:
            length = min(len(seq), total_len)
            tokens[0, :length] = seq[:length]
        elif seq.ndimension() == 2:  # 如果是 2D Tensor，支持批量填充
            for i in range(min(bsz, seq.size(0))):
                length = min(len(seq[i]), total_len)
                tokens[i, :length] = seq[i][:length]
        else:
            raise ValueError("Input seq must be 1D or 2D Tensor.")

        return tokens

# 创建 DataLoader
def get_PoemDataloader(data_dir, batch_size=32, shuffle=True):
    """
    获取 DataLoader
    :param data_dir: 数据文件夹路径
    :param batch_size: 批量大小
    :param shuffle: 是否打乱数据
    :param num_workers: 数据加载的线程数
    :return: PyTorch DataLoader
    """
    dataset = PoemDataset(data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 使用示例
if __name__ == "__main__":
    data_dir = "/inspire/hdd/ws-7c23bd1d-9bae-4238-803a-737a35480e18/aiinfra/yrr-145/xuesheng145-student145/dataset/Tangshi"  # 替换为实际路径
    dataloader = get_PoemDataloader(data_dir, batch_size=4)
    for batch in dataloader:
        input_data = batch["input_ids"]
        target = batch["labels"]
        
        break