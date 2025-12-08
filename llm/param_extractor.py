"""
파라미터 추출 유틸리티
"""
import re
from typing import Dict, Any

# 숫자 한글 → 숫자 매핑
NUMBERS_KR_TO_INT = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10
}

def extract_params(text: str, skill: str) -> Dict[str, Any]:
    """텍스트에서 스킬 파라미터 추출"""
    params = {}
    
    if skill == "walk":
        params = extract_walk_params(text)
    elif skill == "run":
        params = extract_run_params(text)
    elif skill == "turn":
        params = extract_turn_params(text)
    elif skill == "sit":
        params = extract_sit_params(text)
    elif skill == "jump":
        params = extract_jump_params(text)
    # stand, stop, recover_balance는 파라미터 없음
    
    return params

def extract_walk_params(text: str) -> Dict[str, Any]:
    """보행 파라미터 추출"""
    params = {}
    
    # 방향 추출
    if "앞으로" in text or "전진" in text:
        params["direction"] = "forward"
    elif "뒤로" in text or "후퇴" in text or "물러" in text:
        params["direction"] = "backward"
    elif "왼쪽" in text:
        params["direction"] = "left"
    elif "오른쪽" in text:
        params["direction"] = "right"
    else:
        params["direction"] = "forward"  # 기본값
    
    # 걸음 수 추출
    # 한글 숫자 매칭
    for kr_num, int_num in NUMBERS_KR_TO_INT.items():
        if kr_num in text:
            params["steps"] = int_num
            break
    
    # 걸음 패턴 매칭
    if "steps" not in params:
        steps_match = re.search(r'(\d+)걸음', text)
        if steps_match:
            params["steps"] = int(steps_match.group(1))
    
    # 기본값
    if "steps" not in params:
        params["steps"] = 1
    
    return params

def extract_run_params(text: str) -> Dict[str, Any]:
    """달리기 파라미터 추출"""
    params = {}
    
    # 방향 추출
    if "앞으로" in text:
        params["direction"] = "forward"
    elif "뒤로" in text:
        params["direction"] = "backward"
    else:
        params["direction"] = "forward"
    
    # 시간 추출
    duration_match = re.search(r'(\d+)초', text)
    if duration_match:
        params["duration"] = int(duration_match.group(1))
    else:
        params["duration"] = 1
    
    return params

def extract_turn_params(text: str) -> Dict[str, Any]:
    """회전 파라미터 추출"""
    params = {}
    
    # 방향 추출
    if "왼쪽" in text:
        params["direction"] = "left"
    elif "오른쪽" in text:
        params["direction"] = "right"
    else:
        params["direction"] = "left"  # 기본값
    
    # 각도 추출
    angle_match = re.search(r'(\d+)도', text)
    if angle_match:
        angle = int(angle_match.group(1))
        # 표준 각도로 매핑
        if 0 <= angle < 135:
            params["angle"] = 90
        elif 135 <= angle < 225:
            params["angle"] = 180
        elif 225 <= angle < 315:
            params["angle"] = 270
        else:
            params["angle"] = 360
    elif "반바퀴" in text:
        params["angle"] = 180
    elif "한 바퀴" in text or "한바퀴" in text:
        params["angle"] = 360
    else:
        params["angle"] = 90  # 기본값
    
    return params

def extract_sit_params(text: str) -> Dict[str, Any]:
    """앉기 파라미터 추출"""
    params = {}
    
    # 시간 추출
    duration_match = re.search(r'(\d+)초', text)
    if duration_match:
        params["duration"] = int(duration_match.group(1))
    else:
        params["duration"] = 1
    
    return params

def extract_jump_params(text: str) -> Dict[str, Any]:
    """점프 파라미터 추출"""
    params = {}
    
    # 높이 추출
    if any(word in text for word in ["낮게", "살짝", "조금", "폴짝"]):
        params["height"] = "low"
    elif any(word in text for word in ["높게", "크게", "세게", "강하게", "최대한"]):
        params["height"] = "high"
    else:
        params["height"] = "medium"  # 기본값
    
    return params

