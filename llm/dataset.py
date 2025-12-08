"""
스킬 데이터셋 로더
"""
import json
import torch
from torch.utils.data import Dataset
from typing import Dict

class SkillDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, skill_to_idx: Dict[str, int]):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.skill_to_idx = skill_to_idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"], padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "skill_id": torch.tensor(self.skill_to_idx[item["skill"]], dtype=torch.long)
        }

