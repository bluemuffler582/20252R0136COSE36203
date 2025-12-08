"""
자연어 → 스킬 분류 모델
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Iterable


class SkillLanguageModel(nn.Module):
    """자연어 명령을 스킬로 분류하는 언어모델"""
    def __init__(
        self,
        skill_names: List[str],
        hidden_dim: int = 768,
        rejection_threshold: Optional[float] = None,
        supported_skills: Optional[Iterable[str]] = None,
    ):
        """
        Args:
            skill_names: 학습에 사용된 스킬 이름 리스트 (8개)
            hidden_dim: KLUE-BERT hidden dimension (기본 768)
            rejection_threshold:
                - None: confidence 기반 거부 사용 안 함
                - float: 예) 0.5 → confidence < 0.5 이면 거부
            supported_skills:
                - None: 모든 스킬 허용
                - Iterable[str]: 로봇이 실제로 수행 가능한 스킬 집합
        """
        super().__init__()
        self.skill_names = skill_names
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(skill_names)}
        self.idx_to_skill = {idx: skill for skill, idx in self.skill_to_idx.items()}

        self.rejection_threshold = rejection_threshold
        # supported_skills가 None이면 "모든 스킬 허용"
        self.supported_skills = (
            set(supported_skills) if supported_skills is not None else None
        )
        
        # KLUE-BERT 백본
        self.backbone = AutoModel.from_pretrained("klue/bert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        
        # 스킬 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(skill_names))
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        skill_logits = self.classifier(pooled_output)
        return skill_logits
    
    def predict(self, text: str, device: torch.device = None) -> Dict:
        """텍스트 입력으로 스킬 예측 및 거부 판단"""
        self.eval()
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            # 디바이스로 이동
            if device is not None:
                encoded["input_ids"] = encoded["input_ids"].to(device)
                encoded["attention_mask"] = encoded["attention_mask"].to(device)
            
            skill_logits = self.forward(encoded["input_ids"], encoded["attention_mask"])
            skill_probs = torch.softmax(skill_logits, dim=-1)
            skill_idx = torch.argmax(skill_probs, dim=-1).item()
            confidence = skill_probs[0][skill_idx].item()

            predicted_skill = self.idx_to_skill[skill_idx]
            rejected = False

            # 1단계: confidence 기반 거부 (OOD 필터)
            if (
                self.rejection_threshold is not None
                and confidence < self.rejection_threshold
            ):
                predicted_skill = "reject"
                rejected = True

            # 2단계: 로봇이 지원하지 않는 스킬 거부
            if (
                not rejected
                and self.supported_skills is not None
                and predicted_skill not in self.supported_skills
            ):
                predicted_skill = "reject"
                rejected = True
            
            return {
                "skill": predicted_skill,
                "confidence": confidence,
                "rejected": rejected,
            }
