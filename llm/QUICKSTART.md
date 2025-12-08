# 빠른 시작 가이드

## 1단계: 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

## 2단계: 데이터 확인

```bash
# 데이터 이미 생성되어 있음
python test_model.py
```

출력 예시:
```
✓ Loaded 712 samples
Skill distribution:
  walk: 380
  run: 120
  ...
```

## 3단계: 모델 학습

```bash
python train.py --batch_size 16 --epochs 5 --lr 2e-5
```

학습 진행 상황:
- 훈련/검증 손실 및 정확도 표시
- 최고 성능 모델이 `checkpoints/best_model.pt`에 저장됨

## 4단계: 평가

```bash
# 언어모델만 평가
python evaluate.py

# 키워드 라우터와 성능 비교
python evaluate.py --compare
```

## 5단계: 추론

```bash
python predict.py --text "앞으로 다섯 걸음 걸어"
```

예상 출력:
```
Input: 앞으로 다섯 걸음 걸어
Skill: walk
Confidence: 95.23%
```

## 문제 해결

### CUDA/GPU 메모리 부족
- `--batch_size`를 더 작게 설정 (예: 8, 4)
- CPU 사용 시 자동 처리됨

### KLUE 모델 다운로드 실패
- 인터넷 연결 확인
- Hugging Face 로그인 필요할 수 있음

## 다음 단계

- [ ] 파라미터 예측 모듈 추가
- [ ] 스킬 시퀀스 전체 처리
- [ ] 실제 로봇 제어와 통합

