"""
모델 테스트 스크립트 (간단한 동작 확인)
"""
import json

def test_data_loading():
    """데이터 로딩 테스트"""
    print("Testing data loading...")
    with open("data/train_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples")
    
    # 샘플 출력
    print("\nSample data:")
    for i, item in enumerate(data[:3]):
        print(f"{i+1}. {item}")
    
    return True

def test_data_distribution():
    """데이터 분포 확인"""
    print("\nAnalyzing data distribution...")
    with open("data/train_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 스킬별 분포
    skill_counts = {}
    for item in data:
        skill = item["skill"]
        skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    print("\nSkill distribution:")
    for skill, count in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {skill}: {count}")
    
    return True

def main():
    print("=" * 60)
    print("Language Model Test Suite")
    print("=" * 60)
    
    try:
        test_data_loading()
        test_data_distribution()
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

