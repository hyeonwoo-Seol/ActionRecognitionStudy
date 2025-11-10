# analyze_dataset.py
import os
import torch
import numpy as np
import config  # config.py 파일에서 설정을 불러옵니다.
from tqdm import tqdm # 진행 상황을 보여주기 위한 라이브러리

# --- 설정 ---
DATA_DIR = config.DATASET_PATH 
INTERACTION_CLASS_IDS = [f"A{i:03d}" for i in range(50, 61)] # A050 ~ A060

def analyze_interaction_data(data_dir, class_ids):
    """
    지정된 디렉토리에서 상호작용 클래스 파일 전체를 스캔하여
    Person 2 데이터의 유효성을 통계냅니다.
    """
    
    print(f"Scanning for files in: {data_dir}")
    
    try:
        all_files = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found at '{data_dir}'")
        return

    # 1. 상호작용 클래스 ID(A050~A060)를 포함하는 .pt 파일만 필터링
    interaction_files = [
        f for f in all_files 
        if any(class_id in f for class_id in class_ids) and f.endswith('.pt')
    ]
    
    if not interaction_files:
        print(f"Error: No interaction class files (A050-A060) found in {data_dir}")
        return

    print(f"Found {len(interaction_files)} interaction files to analyze...")

    # 2. 통계를 위한 변수 초기화
    total_checked_count = 0
    invalid_p2_count = 0
    invalid_files_list = [] # 문제가 있는 파일 목록

    # 3. tqdm을 사용하여 전체 파일 스캔 (시각화 없이)
    for filename in tqdm(interaction_files, desc="[Analyzing Data]"):
        file_path = os.path.join(data_dir, filename)
        
        try:
            data = torch.load(file_path, map_location='cpu')
            coords = data['first_frame_coords'].cpu().numpy() #
            
            if coords.shape != (50, 3):
                continue # 혹시 모를 shape 오류 파일은 건너뛰기

            # Person 1과 Person 2의 좌표 분리
            p2_coords = coords[25:50, :] #

            # (중요) Person 2 데이터가 유효한지 확인 (0이 아닌 값이 하나라도 있는지)
            p2_is_valid = np.any(p2_coords != 0) #
            
            total_checked_count += 1
            
            if not p2_is_valid:
                invalid_p2_count += 1
                invalid_files_list.append(filename)

        except Exception as e:
            print(f"\nError processing file {filename}: {e}")
            continue

    # 4. 최종 통계 리포트 출력
    print("\n--- 📊 데이터셋 분석 완료 ---")
    print(f"분석 대상 클래스: A050 ~ A060")
    print(f"총 분석 파일 수: {total_checked_count} 개")
    
    valid_count = total_checked_count - invalid_p2_count
    valid_percent = (valid_count / total_checked_count * 100) if total_checked_count > 0 else 0
    invalid_percent = (invalid_p2_count / total_checked_count * 100) if total_checked_count > 0 else 0

    print(f"✅ Person 2 데이터 유효: {valid_count} 개 ({valid_percent:.2f}%)")
    print(f"❌ Person 2 데이터 모두 0 (문제 의심): {invalid_p2_count} 개 ({invalid_percent:.2f}%)")

    # 5. (선택) 문제가 있는 파일 목록 출력
    if invalid_files_list:
        print("\n--- ❌ Person 2가 0인 파일 목록 ---")
        for i, filename in enumerate(invalid_files_list):
            print(filename)
            if i > 20: # 너무 많으면 20개까지만 출력
                print(f"... and {len(invalid_files_list) - 20} more files.")
                break

if __name__ == "__main__":
    analyze_interaction_data(DATA_DIR, INTERACTION_CLASS_IDS)
