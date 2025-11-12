import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

# ## --------------------------------------------------------------------------
# ## --- 설정값 (preprocess_ntu_data.py 참고) ---
# ## --------------------------------------------------------------------------

# >> 처리할 NTU_RGB+D 60 skeleton 데이터가 있는 위치
SOURCE_DATA_PATH = '../paper-review/Action_Recognition/Code/nturgbd01/' 

# >> preprocess_ntu_data.py (config) 에서 사용하는 최대 프레임 (검증용)
MAX_FRAMES_CONFIG = 300 

# ## --------------------------------------------------------------------------
# ## --- 멀티프로세싱 워커 함수 ---
# ## --------------------------------------------------------------------------

def get_frame_count(filename):
    """
    [워커 함수] .skeleton 파일 하나를 열어 첫 줄의 프레임 수를 반환합니다.
    """
    if not filename.endswith('.skeleton'):
        return None

    filepath = os.path.join(SOURCE_DATA_PATH, filename)
    
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            num_frames = int(first_line.strip())
            return num_frames
    except (FileNotFoundError, ValueError, IOError, TypeError):
        # 파일이 비어있거나, 첫 줄이 숫자가 아닌 경우
        # print(f"Warning: Could not read frame count from {filename}")
        return None

# ## --------------------------------------------------------------------------
# ## --- 메인 분석 함수 ---
# ## --------------------------------------------------------------------------

def analyze_video_lengths(data_path):
    """
    [메인 함수] 모든 .skeleton 파일의 원본 프레임 수를 병렬로 집계하고
    분포표(CSV)와 히스토그램(PNG)을 저장합니다.
    """
    all_frame_counts = []
    
    print(f"'{data_path}' 경로의 스켈레톤 파일 분석을 시작합니다...")
    
    try:
        filenames = os.listdir(data_path)
        skeleton_files = [f for f in filenames if f.endswith('.skeleton')]
        if not skeleton_files:
            print(f"오류: '{data_path}' 경로에서 .skeleton 파일을 찾을 수 없습니다.")
            print("스크립트 상단의 'SOURCE_DATA_PATH' 변수를 확인해주세요.")
            return None
    except FileNotFoundError:
        print(f"오류: 디렉터리 경로를 찾을 수 없습니다: '{data_path}'.")
        print("스크립트 상단의 'SOURCE_DATA_PATH' 변수를 확인해주세요.")
        return None

    # --- 멀티프로세싱 시작 ---
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for frame count calculation...")

    with Pool(processes=num_cores) as pool:
        results_iterator = pool.imap_unordered(get_frame_count, skeleton_files)
        
        for frame_count in tqdm(results_iterator, total=len(skeleton_files), desc="파일 처리 중"):
            if frame_count is not None and frame_count > 0:
                all_frame_counts.append(frame_count)
    # --- 멀티프로세싱 종료 ---
                    
    if not all_frame_counts:
        print("오류: 유효한 프레임 수를 가진 스켈레톤 파일을 찾지 못했습니다.")
        return None
        
    print(f"\n총 {len(all_frame_counts)}개의 유효한 스켈레톤 파일에서 프레임 수를 집계했습니다.")
    
    # 1. Pandas Series로 변환하여 통계 분석
    counts_series = pd.Series(all_frame_counts)
    
    # 2. 통계 요약 테이블 생성
    # (90%, 95%, 99% 백분위수 포함)
    distribution_table = counts_series.describe(percentiles=[.25, .5, .75, .90, .95, .99])
    
    print("\n--- 원본 비디오 프레임 수 분포 요약 ---")
    print(distribution_table)
    
    # 3. 통계표를 CSV 파일로 저장
    table_filename = "frame_count_distribution.csv"
    try:
        distribution_table.to_csv(table_filename)
        print(f"\n분포 요약표가 '{table_filename}' 파일로 저장되었습니다.")
    except IOError as e:
        print(f"오류: CSV 파일 저장 실패: {e}")
    
    # 4. 히스토그램 생성 및 PNG 파일로 저장
    plot_filename = "frame_count_histogram.png"
    try:
        plt.figure(figsize=(12, 7))
        
        # 최대값을 기준으로 bin 범위를 설정 (예: 0~600 프레임)
        # 99% 지점 + 100을 최대값으로 하여 너무 긴 꼬리는 자르고 봅니다.
        max_val = int(counts_series.quantile(0.99)) + 100
        min_val = 0
        
        plt.hist(counts_series, bins=range(min_val, max_val, 5), edgecolor='black', alpha=0.7)
        
        # MAX_FRAMES_CONFIG (300) 위치에 수직선 표시
        plt.axvline(MAX_FRAMES_CONFIG, color='red', linestyle='dashed', linewidth=2, 
                    label=f'MAX_FRAMES Cutoff = {MAX_FRAMES_CONFIG}')
        
        median_val = counts_series.median()
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1, 
                    label=f'Median Frame Count: {median_val:.0f}')
        
        plt.title(f'Video Length Distribution (0 - {max_val} Frames)', fontsize=15)
        plt.xlabel('Original Frame Count (Length of Video)', fontsize=12)
        plt.ylabel('Frequency (Number of Videos)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(min_val, max_val) # X축 범위 설정
        
        plt.savefig(plot_filename)
        print(f"히스토그램이 '{plot_filename}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"오류: 히스토그램 저장 실패: {e}")

    return distribution_table, plot_filename

if __name__ == '__main__':
    # 메인 분석 함수 호출
    analyze_video_lengths(SOURCE_DATA_PATH)
