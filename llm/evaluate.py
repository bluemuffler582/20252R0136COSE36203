"""
평가 및 비교 스크립트
"""
import os
import json
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models import SkillLanguageModel
from dataset import SkillDataset

class KeywordRouter:
    """키워드 기반 라우터 (베이스라인)"""
    def __init__(self):
        self.keywords = {
            "walk": ["걸어", "가", "걸음", "걷", "보행"],
            "run": ["달려", "뛰어", "빠르게", "달리"],
            "turn": ["회전", "돌아", "방향"],
            "sit": ["앉아", "앉", "착석"],
            "stand": ["서", "일어서", "입자세", "설"],
            "jump": ["점프", "뛰어", "깡충"],
            "stop": ["멈춰", "정지", "그만"],
            "recover_balance": ["균형", "밸런스", "안정화"]
        }
    
    def predict(self, text: str) -> str:
        """키워드 매칭으로 스킬 예측"""
        text_lower = text.lower()
        scores = {}
        
        for skill, keywords in self.keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[skill] = score
        
        if max(scores.values()) == 0:
            return "walk"  # 기본값
        return max(scores, key=scores.get)

def evaluate_model(model, dataloader, device):
    """모델 평가"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            skill_ids = batch["skill_id"].to(device)
            
            skill_logits = model(input_ids, attention_mask)
            _, predicted = torch.max(skill_logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(skill_ids.cpu().numpy())
    
    return all_preds, all_labels

def evaluate_router(router, dataset):
    """라우터 평가"""
    all_preds = []
    all_labels = []
    skill_to_idx = dataset.skill_to_idx
    
    for i in range(len(dataset)):
        item = dataset.data[i]
        text = item["text"]
        
        # 라벨
        if "skill" in item:
            skill = item["skill"]
        elif "skills" in item:
            skill = item["skills"][0]["skill"]
        else:
            continue
        
        skill_id = skill_to_idx[skill]
        
        # 예측
        pred_skill = router.predict(text)
        pred_id = skill_to_idx[pred_skill]
        
        all_preds.append(pred_id)
        all_labels.append(skill_id)
    
    return all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train_data.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compare", action="store_true", help="키워드 라우터와 비교")
    args = parser.parse_args()
    
    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 스킬 이름
    skill_names = [
        "walk", "run", "turn", "sit", "stand", 
        "jump", "stop", "recover_balance"
    ]
    skill_to_idx = {skill: idx for idx, skill in enumerate(skill_names)}
    
    # 데이터셋
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    dataset = SkillDataset(args.data_path, tokenizer, skill_to_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 로드
    print(f"Loading model from {args.checkpoint}...")
    model = SkillLanguageModel(skill_names=skill_names)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # 평가
    print("Evaluating language model...")
    preds, labels = evaluate_model(model, dataloader, device)
    
    accuracy = accuracy_score(labels, preds)
    print(f"\nLanguage Model Accuracy: {accuracy * 100:.2f}%")
    
    # 분류 리포트
    print("\nClassification Report:")
    print(classification_report(
        labels, preds, 
        target_names=skill_names,
        digits=3
    ))
    
    # 키워드 라우터와 비교
    if args.compare:
        print("\n" + "="*60)
        print("Comparing with Keyword Router...")
        print("="*60)
        
        router = KeywordRouter()
        router_preds, router_labels = evaluate_router(router, dataset)
        
        router_accuracy = accuracy_score(router_labels, router_preds)
        print(f"\nKeyword Router Accuracy: {router_accuracy * 100:.2f}%")
        print(f"Language Model Accuracy: {accuracy * 100:.2f}%")
        print(f"Improvement: {(accuracy - router_accuracy) * 100:.2f}%")
        
        print("\nKeyword Router Classification Report:")
        print(classification_report(
            router_labels, router_preds,
            target_names=skill_names,
            digits=3
        ))

if __name__ == "__main__":
    main()

