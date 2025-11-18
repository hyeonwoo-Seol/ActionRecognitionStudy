import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# ## --------------------------------------------------------------------------
# ## --- 설정값 (config.py 및 preprocess_ntu_data.py 참고) ---
# ## --------------------------------------------------------------------------

# .pt 파일이 저장된 경로
DATASET_PATH = './nturgbd_processed_allNew200/'
# .npz 통계 파일 경로 (preprocess_ntu_data.py와 동일해야 함)
STATS_FILE = './stats_allNew200.npz'
# 훈련 데이터 필터링을 위한 리스트 (preprocess_ntu_data.py와 동일)
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
NUM_COORDS = 15

# 히스토그램을 그릴 범위 (정규화된 값 기준)
BINS = np.linspace(-5.0, 5.0, 201) # -5.0에서 +5.0까지 200개의 구간

# 15개 특징의 이름 (preprocess_ntu_data.py의 _calculate_features 순서)
FEATURE_NAMES = [
    "Dist (Dynamic)",    # 0
    "Dir_X (Dynamic)",   # 1
    "Dir_Y (Dynamic)",   # 2
    "Dir_Z (Dynamic)",   # 3
    "Bone Length (Static)",  # 4
    "Joint Angle (Static)",# 5
    "Rel Angle Y (Static)",# 6
    "Rel Angle Z (Static)",# 7
    "Inter-Dist 1 (Static)",# 8
    "Inter-Dist 2 (Static)",# 9
    "P0-P1 Center (Inter)", # 10
    "P0-P1 RH-RH (Inter)", # 11
    "P0-P1 LH-LH (Inter)", # 12
    "P0-P1 RF-RF (Inter)", # 13
    "P0-P1 LF-LF (Inter)"  # 14
]

# ## --------------------------------------------------------------------------
# ## --- 멀티프로세싱 워커 함수 ---
# ## --------------------------------------------------------------------------

def process_file_for_hist(filename):
    """
    [워커 함수] .pt 파일 하나를 읽어, 정규화한 뒤, 15개 특징의 히스토그램을 반환합니다.
    """
    try:
        # 1. 훈련용 데이터인지 확인
        subject_id = int(filename[9:12]) # S001P001... -> P
        if subject_id not in TRAINING_SUBJECTS:
            return None
            
        # 2. 통계 파일 로드 (워커마다 로드해야 함)
        if not hasattr(process_file_for_hist, 'mean_std'):
            stats = np.load(STATS_FILE)
            mean = torch.from_numpy(stats['mean'].flatten()).float()
            std = torch.from_numpy(stats['std'].flatten()).float()
            std_eps = std + 1e-8 # 0으로 나누기 방지
            process_file_for_hist.mean_std = (mean, std_eps)
        
        mean, std_eps = process_file_for_hist.mean_std
        
        # 3. .pt 파일 로드
        filepath = os.path.join(DATASET_PATH, filename)
        data = torch.load(filepath, map_location='cpu')
        features = data['data'] # (T, J, C) = (100, 50, 15)
        
        # 4. 유효한 프레임만 추출 (0으로 패딩된 부분 제외)
        # (T*J, C)
        features_flat = features.reshape(-1, NUM_COORDS)
        # 0이 아닌 프레임/관절만 필터링 (preprocess_ntu_data.py의 방식과 유사)
        valid_mask = torch.abs(features_flat).sum(dim=1) > 1e-6
        valid_features = features_flat[valid_mask]
        
        if valid_features.shape[0] == 0:
            return None
            
        # 5. 정규화 수행 (ntu_data_loader.py와 동일)
        normalized_features = (valid_features - mean) / std_eps
        
        # 6. 15개 특징 각각에 대해 히스토그램 계산
        hist_counts_list = []
        for i in range(NUM_COORDS):
            # np.histogram은 (counts, bin_edges)를 반환
            counts, _ = np.histogram(normalized_features[:, i].numpy(), bins=BINS)
            hist_counts_list.append(counts)
            
        return hist_counts_list
        
    except Exception as e:
        # print(f"Warning: Failed processing {filename}. Error: {e}")
        return None

# ## --------------------------------------------------------------------------
# ## --- 메인 분석 및 플로팅 함수 ---
# ## --------------------------------------------------------------------------

def plot_histograms(total_histograms, bin_edges, feature_names):
    """15개의 히스토그램을 5x3 그리드로 저장합니다."""
    print("Plotting histograms...")
    
    fig, axes = plt.subplots(5, 3, figsize=(20, 25))
    fig.suptitle('Distribution of Normalized Features (Z-scores)', fontsize=20, y=1.02)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    for i in range(NUM_COORDS):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        counts = total_histograms[i]
        
        # 로그 스케일 적용 (분포가 너무 뾰족할 경우 대비)
        # 0인 빈을 그리기 위해 +1
        counts_log = np.log1p(counts) 
        
        ax.bar(bin_centers, counts_log, width=bin_width, align='center', alpha=0.8)
        
        ax.set_title(f"Feature {i}: {feature_names[i]}")
        ax.set_xlabel("Normalized Value (Z-score)")
        ax.set_ylabel("Log(Frequency + 1)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 평균(0) 위치에 빨간 선 표시
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Mean (0.0)')
        # +/- 1, 2 표준편차 위치 표시
        ax.axvline(-1, color='green', linestyle=':', linewidth=1)
        ax.axvline(1, color='green', linestyle=':', linewidth=1)
        ax.axvline(-2, color='gray', linestyle=':', linewidth=1)
        ax.axvline(2, color='gray', linestyle=':', linewidth=1)
        
        ax.set_xlim(BINS[0], BINS[-1]) # -5.0 ~ 5.0
        ax.legend(loc='upper right')

    plt.tight_layout()
    plot_filename = "feature_distribution_histograms.png"
    plt.savefig(plot_filename, dpi=100)
    print(f"Feature distribution histograms saved to '{plot_filename}'")
    plt.close()

def analyze_feature_distribution():
    """
    [메인 함수] 훈련용 .pt 파일들을 병렬로 읽어 15개 특징의 분포를 시각화합니다.
    """
    
    # 1. 통계 파일 존재 여부 확인
    if not os.path.exists(STATS_FILE):
        print(f"오류: 통계 파일 '{STATS_FILE}'을 찾을 수 없습니다.")
        print("먼저 preprocess_ntu_data.py의 calculate_and_save_stats()를 실행해야 합니다.")
        return
    print(f"Using stats file: '{STATS_FILE}'")

    # 2. .pt 파일 목록 로드 (훈련용만 필터링 준비)
    try:
        all_filenames = [f for f in os.listdir(DATASET_PATH) if f.endswith('.pt')]
        # 훈련용 파일만 필터링
        training_files = [
            f for f in all_filenames 
            if int(f[9:12]) in TRAINING_SUBJECTS
        ]
        if not training_files:
            print(f"오류: '{DATASET_PATH}'에서 훈련용(.pt) 파일을 찾을 수 없습니다.")
            return
    except (FileNotFoundError, IndexError):
        print(f"오류: '{DATASET_PATH}' 경로를 찾을 수 없거나 파일 이름 형식이 다릅니다.")
        return

    print(f"Analyzing distributions from {len(training_files)} training files...")

    # 3. 멀티프로세싱으로 히스토그램 집계
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    # (15, 200) 크기의 0으로 채워진 리스트 초기화
    total_histograms = [np.zeros(len(BINS) - 1, dtype=np.int64) for _ in range(NUM_COORDS)]

    with Pool(processes=num_cores) as pool:
        results_iterator = pool.imap_unordered(process_file_for_hist, training_files)
        
        for hist_list in tqdm(results_iterator, total=len(training_files), desc="[Analyzing Features]"):
            if hist_list is not None and len(hist_list) == NUM_COORDS:
                # 15개 특징 각각의 히스토그램 카운트를 누적
                for i in range(NUM_COORDS):
                    total_histograms[i] += hist_list[i]

    # 4. 최종 플롯 저장
    plot_histograms(total_histograms, BINS, FEATURE_NAMES)
    
    print("\n분석 완료.")

if __name__ == '__main__':
    analyze_feature_distribution()
