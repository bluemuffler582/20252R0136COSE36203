"""
학습 스크립트
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

from models import SkillLanguageModel
from dataset import SkillDataset

def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        skill_ids = batch["skill_id"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        skill_logits = model(input_ids, attention_mask)
        loss = criterion(skill_logits, skill_ids)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(skill_logits.data, 1)
        total += skill_ids.size(0)
        correct += (predicted == skill_ids).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            skill_ids = batch["skill_id"].to(device)
            
            skill_logits = model(input_ids, attention_mask)
            loss = criterion(skill_logits, skill_ids)
            
            total_loss += loss.item()
            _, predicted = torch.max(skill_logits.data, 1)
            total += skill_ids.size(0)
            correct += (predicted == skill_ids).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train_data.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()
    
    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 스킬 이름 로드
    skill_names = [
        "walk", "run", "turn", "sit", "stand", 
        "jump", "stop", "recover_balance"
    ]
    skill_to_idx = {skill: idx for idx, skill in enumerate(skill_names)}
    
    # 모델 및 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = SkillLanguageModel(skill_names=skill_names)
    model.to(device)
    
    # 데이터셋 로드
    full_dataset = SkillDataset(args.data_path, tokenizer, skill_to_idx)
    
    # Train/Val 분할
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 체크포인트 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 학습 루프
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 평가
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 베스트 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "skill_names": skill_names
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining completed! Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()

