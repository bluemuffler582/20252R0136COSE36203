"""
추론 스크립트
"""
import json
import torch
import argparse
from models import SkillLanguageModel
from param_extractor import extract_params

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = SCRIPT_DIR / "checkpoints" / "best_model.pt"


# confidence 기반 거부 기준
REJECTION_THRESHOLD = 0.5

# 현재 로봇이 실제로 지원하는 스킬
SUPPORTED_SKILLS = ["walk", "turn"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="자연어 명령어")
    #parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CKPT),
        help="Path to model checkpoint",
    )
    parser.add_argument("--output", type=str, choices=["text", "json"], default="json")
    args = parser.parse_args()
    
    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 스킬 이름 (모델이 학습한 전체 스킬 목록)
    skill_names = [
        "walk", "run", "turn", "sit", "stand", 
        "jump", "stop", "recover_balance"
    ]
    
    # 모델 로드
    model = SkillLanguageModel(
        skill_names=skill_names,
        rejection_threshold=REJECTION_THRESHOLD,
        supported_skills=SUPPORTED_SKILLS,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # 예측
    result = model.predict(args.text, device=device)
    skill = result["skill"]
    confidence = result["confidence"]
    rejected = result.get("rejected", False)
    
    # 거부된 경우에는 파라미터 추출하지 않음
    if rejected or skill == "reject":
        params = {}
    else:
        params = extract_params(args.text, skill)
    
    # 출력 형식
    if args.output == "json":
        # JSON 형식
        output = {
            "text": args.text,
            "skill": skill,
            "params": params
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"Input: {args.text}")
        print(f"Skill: {skill}")
        print(f"Params: {params}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Rejected: {rejected}")

if __name__ == "__main__":
    main()
