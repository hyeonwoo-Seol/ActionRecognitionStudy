# check_stats.py
import numpy as np
import os
import time

# config.py, ntu_data_loader.py 등과 동일한 경로를 사용해야 합니다.
STATS_FILE = './stats_allNew200.npz' 

if not os.path.exists(STATS_FILE):
    print(f"오류: '{STATS_FILE}' 파일을 찾을 수 없습니다.")
    print("preprocess_ntu_data.py를 먼저 실행해야 합니다.")
else:
    try:
        # 파일의 최종 수정 시간 확인
        mod_time = os.path.getmtime(STATS_FILE)
        print(f"'{STATS_FILE}' 파일 로드 성공.")
        print(f"파일 최종 수정 시간: {time.ctime(mod_time)}")
        
        # npz 파일 로드
        stats = np.load(STATS_FILE)
        
        # 'mean' 키가 있는지 확인하고 1D로 펼치기
        if 'mean' in stats:
            mean_values = stats['mean'].flatten()
            print("\n--- 로드된 15개 특징의 평균값 ---")
            print(mean_values)
            print("\n-------------------------------------")
            
            # Feature 0과 10의 값을 구체적으로 확인
            print(f"Feature 0 (Dist) Mean: {mean_values[0]:.6f}")
            print(f"Feature 8 (Inter-Dist 1) Mean: {mean_values[8]:.6f}")
            print(f"Feature 10 (P0-P1 Center) Mean: {mean_values[10]:.6f}")

        else:
            print("오류: 파일에 'mean' 데이터가 없습니다.")
            
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
