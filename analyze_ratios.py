import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from mpl_toolkits.mplot3d import Axes3D # _read_skeleton_file이 참조할 수 있으므로 유지
from multiprocessing import Pool, cpu_count # ! 멀티프로세싱을 위해 추가

# ## --------------------------------------------------------------------------
# ## --- 시작: visualize_skeleton.py에서 복사한 코드 ---
# ##
# ## 원본 파일의 경로와 스켈레톤 정의, 파일 읽기 함수를 그대로 가져옵니다.
# ## --------------------------------------------------------------------------

# >> 처리할 NTU_RGB+D 60 skeleton 데이터가 있는 위치
SOURCE_DATA_PATH = '../paper-review/Action_Recognition/Code/nturgbd01/' 

# >> NTU 데이터셋의 기본 관절 수
BASE_NUM_JOINTS = 25

# ## ---------------------------------------------------------------------------------
# 뼈 길이 계산을 위한 관절 연결 정보다. (부모 -> 자식)
# ## ---------------------------------------------------------------------------------
SKELETON_BONES = [
    (20, 1), (1, 0), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
] # 총 24개의 뼈

# ##----------------------------------------------------------------------------------
# NTU_RGB+D 데이터셋의 .skeleton 파일을 읽어 파싱하는 함수이다.
# (preprocess_ntu_data.py의 함수와 동일)
# ##----------------------------------------------------------------------------------
def _read_skeleton_file(filepath):
    # >> 첫 번째 라인에서 전체 프레임 수를 먼저 읽는다.
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            num_frames = int(first_line)
    except (FileNotFoundError, ValueError, IOError):
        # print(f"Error: Could not read frame count from {filepath}")
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    # >> 파일의 첫 줄에서 읽어온 프레임 수가 0인 경우,
    if num_frames == 0:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    # >> 1. 첫 번째 스캔: 실제 데이터가 있는 bodyID 2개 찾기
    body_id_counts = {} # 투표함 준비
    try:
        with open(filepath, 'r') as f:
            f.readline()  # 첫 번째 프레임 수 라인 건너뛰기

            line_idx = 1
            while True: # 파일 전체를 한 프레임씩, 한 사람씩 스캔하기.
                line = f.readline()
                if not line: break # 파일 끝
                line_idx += 1

                try:
                    num_bodies = int(line.strip())
                except ValueError: 
                    # print(f"Warning: Invalid body count at line {line_idx} in {filepath}. Skipping frame.")
                    continue

                for i in range(num_bodies):
                    line = f.readline() # body info line을 읽는다.
                    if not line: break
                    line_idx += 1
                    
                    body_info = line.strip().split() 
                    if len(body_info) < 1: 
                        # print(f"Warning: Skipping empty body info line {line_idx} in {filepath}")
                        continue
                    
                    body_id = body_info[0] 

                    line = f.readline() # 관절 수가 적힌 다음줄을 읽는다.
                    if not line: break
                    line_idx += 1

                    try:
                        num_joints = int(line.strip())
                    except ValueError:
                        # print(f"Warning: Invalid joint count at line {line_idx} in {filepath}.")
                        num_joints = 0
                    
                    has_non_zero_coord = False
                    for j in range(num_joints):
                        line = f.readline()
                        if not line: break
                        line_idx += 1
                        
                        if not has_non_zero_coord:
                            try:
                                joint_info = line.strip().split()
                                if any(float(coord) != 0.0 for coord in joint_info[:3]):
                                    has_non_zero_coord = True
                            except (ValueError, IndexError):
                                continue
                    
                    if has_non_zero_coord:
                        body_id_counts[body_id] = body_id_counts.get(body_id, 0) + 1
    except IOError as e:
        # print(f"Error during Pass 1 scan of {filepath}: {e}")
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    sorted_body_ids = sorted(body_id_counts.items(), key=lambda item: item[1], reverse=True)
    
    body1_id = sorted_body_ids[0][0] if len(sorted_body_ids) > 0 else None
    body2_id = sorted_body_ids[1][0] if len(sorted_body_ids) > 1 else None

    # >> 2. 두 번째 스캔: 결정된 ID를 기준으로 3D 좌표를 추출해서 넘파이 배열을 만든다.
    final_coords = np.zeros((num_frames, 2, BASE_NUM_JOINTS, 3)) 
    try:
        with open(filepath, 'r') as f: # 파일을 처음부터 연다.
            f.readline()  # 첫 번째 프레임 수 라인 건너뛴다.
            f_idx = 0
            line_idx = 1
            while f_idx < num_frames: # 프레임 단위로 순회한다.
                line = f.readline()
                if not line: break
                line_idx += 1
                
                try:
                    num_bodies = int(line.strip())
                except ValueError:
                    continue 

                for i in range(num_bodies):
                    line = f.readline()
                    if not line: break
                    line_idx += 1
                    
                    body_info = line.strip().split() 
                    if len(body_info) < 1: continue

                    current_body_id = body_info[0] 
                    
                    line = f.readline()
                    if not line: break
                    line_idx += 1
                    
                    try:
                        num_joints = int(line.strip())
                    except ValueError:
                        num_joints = 0

                    target_person_idx = -1
                    if current_body_id == body1_id:
                        target_person_idx = 0
                    elif current_body_id == body2_id:
                        target_person_idx = 1
                    
                    for j in range(num_joints):
                        line = f.readline() # joint info line
                        if not line: break
                        line_idx += 1
                        
                        if target_person_idx != -1 and j < BASE_NUM_JOINTS:
                            try:
                                joint_info = line.strip().split()
                                x, y, z = map(float, joint_info[:3])
                                final_coords[f_idx, target_person_idx, j] = [x, y, z]
                            except (ValueError, IndexError):
                                continue
                f_idx += 1
    except IOError as e:
        # print(f"Error during Pass 2 scan of {filepath}: {e}")
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    return final_coords

# ## --------------------------------------------------------------------------
# ## --- 끝: visualize_skeleton.py에서 복사한 코드 ---
# ## --------------------------------------------------------------------------


# ## --------------------------------------------------------------------------
# ## --- 시작: 수정된 비율 분석 코드 (멀티프로세싱 적용) ---
# ## --------------------------------------------------------------------------

def _get_euclidean_distance(p1, p2):
    """두 3D 점 사이의 유클리드 거리를 계산합니다."""
    return np.linalg.norm(p1 - p2)

# --- 관절 인덱스 정의 (NTU 25-joint 기준) ---
# ! 멀티프로세싱 워커 함수가 접근할 수 있도록 전역 범위로 이동
J_SPINE_BASE = 0     # 척추 하단
J_SPINE_MID = 1      # 척추 중간
J_SPINE_SHOULDER = 20  # 어깨 중심
J_RIGHT_SHOULDER = 8   # 오른쪽 어깨
J_RIGHT_ELBOW = 9    # 오른쪽 팔꿈치
J_RIGHT_WRIST = 10   # 오른쪽 손목


def process_file_for_ratios(filename):
    """
    [워커 함수] 파일 하나를 읽어 (오른팔/척추) 비율 리스트를 반환합니다.
    """
    if not filename.endswith('.skeleton'):
        return [] # .skeleton 파일이 아니면 빈 리스트 반환

    filepath = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(filepath) # (T, 2, 25, 3)
    
    if coords.shape[0] == 0:
        return []

    file_ratios = [] # 이 파일에서 계산된 비율을 저장할 리스트
    num_frames = coords.shape[0]
    num_persons = coords.shape[1]
    
    # 2. 모든 프레임, 모든 사람에 대해 반복
    for t in range(num_frames):
        for p in range(num_persons):
            joints_3d = coords[t, p, :, :] # (25, 3)
            
            # 3. 유효한 스켈레톤인지 확인 (어깨 중심점이 0,0,0이 아니면 유효)
            if np.all(joints_3d[J_SPINE_SHOULDER] == 0):
                continue
                
            # 4. 관절 좌표 가져오기
            p_spine_base = joints_3d[J_SPINE_BASE]
            p_spine_mid = joints_3d[J_SPINE_MID]
            p_spine_shoulder = joints_3d[J_SPINE_SHOULDER]
            
            p_right_shoulder = joints_3d[J_RIGHT_SHOULDER]
            p_right_elbow = joints_3d[J_RIGHT_ELBOW]
            p_right_wrist = joints_3d[J_RIGHT_WRIST]
            
            # 5. 뼈 길이 계산
            # 척추 길이: (0->1 거리) + (1->20 거리)
            spine_len = _get_euclidean_distance(p_spine_base, p_spine_mid) + \
                        _get_euclidean_distance(p_spine_mid, p_spine_shoulder)
                        
            # 오른팔 길이: (8->9 거리) + (9->10 거리)
            right_arm_len = _get_euclidean_distance(p_right_shoulder, p_right_elbow) + \
                            _get_euclidean_distance(p_right_elbow, p_right_wrist)
                            
            # 6. 유효한 길이인지 확인 (0으로 나누기 방지)
            if spine_len == 0 or right_arm_len == 0:
                continue
                
            # 7. 비율 계산 및 저장
            ratio = right_arm_len / spine_len
            file_ratios.append(ratio)

    return file_ratios


def analyze_body_part_ratios(data_path):
    """
    [메인 함수] 모든 .skeleton 파일을 병렬로 순회하며 비율을 계산하고
    분포표(CSV)와 히스토그램(PNG)을 저장합니다.
    """
    all_ratios = []
    
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

    # ! --- 멀티프로세싱 시작 ---
    # 사용할 CPU 코어 수를 정합니다.
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for ratio calculation...")

    # 계산 중 발생하는 (e.g., 0/0) 경고를 무시합니다.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # 멀티프로세싱 Pool을 생성하고 작업을 분배합니다.
        with Pool(processes=num_cores) as pool:
            # pool.imap_unordered를 사용하여 작업을 병렬 처리하고 tqdm으로 진행률을 표시합니다.
            results_iterator = pool.imap_unordered(process_file_for_ratios, skeleton_files)
            
            # 반환된 결과(비율 리스트)를 all_ratios에 병합합니다.
            for file_ratios in tqdm(results_iterator, total=len(skeleton_files), desc="파일 처리 중"):
                if file_ratios: # 빈 리스트가 아닐 경우에만
                    all_ratios.extend(file_ratios) # append가 아닌 extend 사용
    # ! --- 멀티프로세싱 종료 ---
                    
    if not all_ratios:
        print("오류: 비율을 계산할 유효한 스켈레톤 데이터를 찾지 못했습니다.")
        return None
        
    print(f"\n총 {len(all_ratios)}개의 유효한 스켈레톤 프레임에서 비율을 계산했습니다.")
    
    # 8. Pandas Series로 변환하여 통계 분석
    ratios_series = pd.Series(all_ratios)
    
    # 9. 통계 요약 테이블 생성
    distribution_table = ratios_series.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    
    print("\n--- (오른팔 길이 / 척추 길이) 비율 분포 요약 ---")
    print(distribution_table)
    
    # 10. 통계표를 CSV 파일로 저장
    table_filename = "body_ratio_distribution.csv"
    try:
        distribution_table.to_csv(table_filename)
        print(f"\n분포 요약표가 '{table_filename}' 파일로 저장되었습니다.")
    except IOError as e:
        print(f"오류: CSV 파일 저장 실패: {e}")
    
    # 11. 히스토그램 생성 및 PNG 파일로 저장
    plot_filename = "body_ratio_histogram.png"
    try:
        plt.figure(figsize=(12, 7))
        
        q_low = ratios_series.quantile(0.01)
        q_high = ratios_series.quantile(0.99)
        filtered_ratios = ratios_series[(ratios_series >= q_low) & (ratios_series <= q_high)]
        
        plt.hist(filtered_ratios, bins=100, edgecolor='black', alpha=0.7)
        
        mean_val = ratios_series.mean()
        median_val = ratios_series.median()
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.3f}')
        
        plt.title('Distribution of (Right Arm Length / Spine Length) Ratio (1st-99th Percentile)', fontsize=15)
        plt.xlabel('Ratio (Right Arm / Spine)', fontsize=12)
        plt.ylabel('Frequency (Count of Skeletons)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(plot_filename)
        print(f"히스토그램이 '{plot_filename}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"오류: 히스토그램 저장 실패: {e}")

    return distribution_table, plot_filename

if __name__ == '__main__':
    # 메인 분석 함수 호출
    analyze_body_part_ratios(SOURCE_DATA_PATH)
