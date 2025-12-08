# 로봇 명령어 언어모델

자연어 명령어를 스킬+인자 구조로 변환하는 경량 언어모델

## 원리 (3줄 요약)

1. **KLUE-BERT**로 한국어 명령을 임베딩 → 스킬 분류 (8개 클래스)
2. 모델이 예측한 스킬에 따라 **규칙 기반 파라미터 추출** (방향, 거리, 각도 등)
3. 키워드 매칭 대비 **+31.5% 성능** 향상

## 프로젝트 구조

```
language_model/
├── data/                    # 데이터셋
│   └── train_data.json      # 학습 데이터
├── models/                  # 모델 정의
│   ├── __init__.py
│   └── skill_model.py       # SkillLanguageModel
├── checkpoints/             # 학습된 모델 (자동 생성)
├── dataset.py              # 데이터셋 로더
├── train.py                # 학습 스크립트
├── evaluate.py             # 평가 스크립트
├── predict.py              # 추론 스크립트
├── test_model.py           # 데이터 테스트
├── requirements.txt        # 의존성
└── README.md
```

## 데이터 형식

입력: 자연어 명령어
```
"앞으로 다섯 걸음 걸어"
"왼쪽으로 90도 회전해"
"3초간 앉아 있어"
```

출력: 스킬 분류
```json
{"skill": "walk", "confidence": 0.95}
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 데이터 확인
```bash
python test_model.py
```

### 2. 모델 학습
```bash
python train.py --batch_size 16 --epochs 10 --lr 2e-5
```

### 3. 평가 및 비교
```bash
# 언어모델 평가
python evaluate.py

# 키워드 라우터와 비교
python evaluate.py --compare
```

### 4. 추론
```bash
python predict.py --text "앞으로 다섯 걸음 걸어"
python predict.py --text "왼쪽으로 90도 회전해"
```

## 아키텍처

- **Backbone**: KLUE-BERT (한국어 pre-trained 모델)
- **Task**: 스킬 분류 (8개 스킬: walk, run, turn, sit, stand, jump, stop, recover_balance)
- **특징**: 경량화된 헤드 레이어로 도메인 특화 학습

## 데이터셋

- **총 241개** 샘플

### 스킬 분포
각 스킬당 30개씩 균등 분포
- walk (보행)
- run (달리기)  
- turn (회전)
- sit (앉기)
- stand (서기)
- jump (점프)
- stop (정지)
- recover_balance (균형 회복)

## 예시

입력: "앞으로 다섯 걸음 걸어"
```json
{
  "text": "앞으로 다섯 걸음 걸어",
  "skill": "walk",
  "params": {
    "direction": "forward",
    "steps": 5
  }
}
```

입력: "왼쪽으로 90도 회전해"
```json
{
  "text": "왼쪽으로 90도 회전해",
  "skill": "turn",
  "params": {
    "direction": "left",
    "angle": 90
  }
}
```

입력: "3초간 앉아 있어"
```json
{
  "text": "3초간 앉아 있어",
  "skill": "sit",
  "params": {
    "duration": 3
  }
}
```

