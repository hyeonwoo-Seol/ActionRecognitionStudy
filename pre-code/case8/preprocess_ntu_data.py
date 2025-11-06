# preprocess_ntu_data.py
# ##--------------------------------------------------------------------------
# x, y, z skeleton 좌표를 이동 거리, 방향, 가속도로 미리 변환하는 파일이다.
# 구현해야 할 기능
# 1. 파일 읽어오기
# 2. 좌표값을 내가 원하는 형태로 변환하기
# 3. 평균과 표준편차를 계산하기
# 4. 이를 통합하는 main 함수 만들기
# ##----------------------------------------------------------------------------

import os
import numpy as np
import torch
from tqdm import tqdm
import config
from multiprocessing import Pool, cpu_count


# >> 처리할 NTU_RGB+D 60 skeleton 데이터가 있는 위치
SOURCE_DATA_PATH = '../../paper-review/Action_Recognition/Code/nturgbd01/' 
# >> 처리가 완료된 파일들을 저장할 위치
TARGET_DATA_PATH = '../nturgbd_processed_allNew/'
# >> 훈련 데이터셋의 평균과 표준편차를 저장할 파일 이름 
STATS_FILE = '../stats_allNew.npz'
# >> 모델에 입력으로 사용할 최대 프레임 수 (config 파일에서 가져옴)
# >> 이보다 길면 잘라내고, 짧으면 0으로 채운다 (padding)
MAX_FRAMES = config.MAX_FRAMES
# >> 모델에서 사용할 총 관절 수 (config 파일에서 가져옴)
# >> NTU 데이터는 최대 2명의 사람을 포함하므로, 25 * 2 = 50이 된다.
NUM_JOINTS = config.NUM_JOINTS
# >> NTU 데이터셋의 기본 관절 수 (한 사람당 25개)
BASE_NUM_JOINTS = 25
# >> 훈련에 사용할 피실험자(subject) ID 목록.
# >> 이 ID를 기준으로 훈련 데이터와 검증/테스트 데이터를 나눕니다.
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

# ## ---------------------------------------------------------------------------------
# >> [수정됨] 뼈 길이 계산을 위한 관절 연결 정보 (부모 -> 자식)
# >> model.py의 get_ntu_shift_decompositions와 동일한 구조
# ## ---------------------------------------------------------------------------------
SKELETON_BONES = [
    (20, 1), (1, 0), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
] # 총 24개의 뼈

# ## ---------------------------------------------------------------------------------
# >> [수정됨] 관절 각도 계산을 위한 (조부모, 부모, 자식) 관절 트리플렛
# ## ---------------------------------------------------------------------------------
JOINT_ANGLE_TRIPLETS = [
    (20, 4, 5), (4, 5, 6), (5, 6, 7),        # 오른팔
    (20, 8, 9), (8, 9, 10), (9, 10, 11),     # 왼팔
    (0, 12, 13), (12, 13, 14), (13, 14, 15),  # 오른다리
    (0, 16, 17), (16, 17, 18), (17, 18, 19),  # 왼다리
    (20, 1, 0), (1, 20, 2), (1, 20, 4), (1, 20, 8), # 몸통/목
    (12, 0, 16), (4, 20, 8)                   # 골반/어깨 너비
] # 총 20개의 각도




# ##----------------------------------------------------------------------------------
# NTU_RGB+D 데이터셋의 .skeleton 파일을 읽어 파싱하는 함수이다.
# 파일을 읽어 (num_frames, 2, 25, 3) 형태의 3D 좌표 numpy 배열로 반환한다.
# ##----------------------------------------------------------------------------------
def _read_skeleton_file(filepath):
    # >> 파일을 열어서 라인들을 읽어온다.
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))
    

    # >> 첫 번째 라인은 전체 프레임 수를 나타낸다.
    num_frames = int(lines[0])
    frames_data = [] # 각 프레임의 데이터를 저장할 리스트
    line_idx = 1 # 현재 읽고 있는 라인 수


    # >> 전체 프레임 수만큼 반복한다.
    for _ in range(num_frames):
        if line_idx >= len(lines): break # 파일 끝에 도달하면 중단
        
        # >> 해당 프레임에 감지된 사람(body)의 수를 읽는다.
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1
        
        # >> 처음에 각 프레임은 (2, 25, 3) shape의 0으로 채워진 배열로 초기화된다.
        frame_person_coords = np.zeros((2, BASE_NUM_JOINTS, 3))


        # >> 감지된 사람 수만큼 반복한다.
        for i in range(num_bodies):
            if line_idx >= len(lines): break
            
            # bodyID, clipped 등의 정보가 담긴 라인을 읽고 다음 라인으로 넘어간다.
            # 사용 안 하기 때문이다.
            line_idx += 1 
            
            if line_idx >= len(lines): break
            # >> 해당 사람의 관절(joint) 수를 읽는다.
            num_joints = int(lines[line_idx].strip())
            line_idx += 1

            # >> 관절 수만큼 반복하여 각 관절의 좌표를 읽는다.
            for j in range(num_joints):
                if line_idx >= len(lines): break
                
                # >> 한 라인에 있는 관절 정보를 공백 기준으로 분리한다.
                joint_info = lines[line_idx].strip().split()
                line_idx += 1
                
                # >> 최대 2명의 사람(i < 2)과 25개의 관절(j < BASE_NUM_JOINTS) 정보만 저장한다.
                if i < 2 and j < BASE_NUM_JOINTS:
                    x, y, z = map(float, joint_info[:3]) # x, y, z 좌표만 추출하고 float으로 변환
                    frame_person_coords[i, j] = [x, y, z] # 미리 만들어둔 배열에 좌표를 저장
        
        # >> 현재 프레임의 좌표 데이터를 리스트에 추가한다.
        frames_data.append(frame_person_coords)
    

    # >> 만약 처리된 프레임 데이터가 없다면 빈 배열을 반환한다.
    if not frames_data:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))


    # >>파이썬 리스트를 numpy 배열로 변환하여 반환한다.
    return np.stack(frames_data)




# ## --------------------------------------------------------------------------------
# 사람마다 다른 신체 크기 영향을 줄이기 위해 척추 길이를 기준으로 3D 스켈레톤 좌표를 정규화합니다.
# 모든 좌표를 척추 길이로 나누어 스켈레톤의 상대적인 크기를 일정하게 만듭니다.
# ## --------------------------------------------------------------------------------
def _normalize_by_bone_length(coords):
    # >> 기준 관절 인덱스를 정의한다.
    SPINE_SHOULDER_JOINT = 20 # 척추 상단 (어깨 중심)
    SPINE_BASE_JOINT = 0 # 척추 하단 (골반 중심)
    

    # >> 각 프레임, 각 사람(person)에 대해 기준 관절의 좌표를 선택합니다.
    # >> shape: (num_frames, 2, 1, 3)
    ref_joint1_coords = coords[:, :, SPINE_SHOULDER_JOINT:SPINE_SHOULDER_JOINT+1, :]
    ref_joint2_coords = coords[:, :, SPINE_BASE_JOINT:SPINE_BASE_JOINT+1, :]
    

    # >> 척추 길이(유클리드 거리)를 계산합니다. 0으로 나누는 것을 방지하기 위해 작은 값(epsilon)을 더한다.
    # >> shape: (num_frames, 2, 1)
    torso_lengths = np.linalg.norm(ref_joint1_coords - ref_joint2_coords, axis=-1) + 1e-8


    # >> 모든 좌표에 나누기 연산을 적용하기 위해 브로드캐스팅이 가능하도록 차원을 추가한다.
    # >> (num_frames, 2, 1) -> (num_frames, 2, 1, 1)
    torso_lengths = np.expand_dims(torso_lengths, axis=-1)
    

    # >> 모든 관절 좌표를 해당 프레임의 척추 길이로 나눈다.
    normalized_coords = coords / torso_lengths
    
    return normalized_coords




# ## ------------------------------------------------------------------------------
# [수정됨] 좌표로부터 동적 특징(7D)과 정적 특징(2D)을 계산한다.
# shape: (num_frames, 50, 9)
# (50 = 25관절 * 2명, 9 = 거리(1) + 방향(3) + 가속도(3) + 뼈길이(1) + 관절각도(1))
# ## ------------------------------------------------------------------------------
def _calculate_features(coords):
    # >> coords shape: (T, 2, 25, 3)
    T = coords.shape[0]
    if T == 0:
        # >> 빈 프레임이면 (0, 50, 9) 형태의 빈 배열 반환
        return np.zeros((0, NUM_JOINTS, config.NUM_COORDS))

    # --- 1. (기존) 동적 특징 계산 (7D) ---
    
    # >> 스켈레톤 중심화
    center_joint = coords[:, :, 0:1, :]
    centered_coords = coords - center_joint

    # >> 변위 계산
    displacement = np.zeros_like(centered_coords)
    displacement[1:] = centered_coords[1:] - centered_coords[:-1]
    
    # >> 거리 계산
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)

    # >> 방향 계산
    magnitude_disp = np.linalg.norm(displacement, axis=-1, keepdims=True) + 1e-8
    direction = displacement / magnitude_disp

    # >> 가속도 계산
    acceleration = np.zeros_like(displacement)
    acceleration[1:] = displacement[1:] - displacement[:-1]

    # >> 동적 특징 결합 (T, 2, 25, 7)
    dynamic_features = np.concatenate((distance, direction, acceleration), axis=-1)


    # --- 2. [신규] 정적 특징 계산 (2D) ---
    
    # >> 2-1. 뼈 길이 특징 (1D)
    # >> 뼈 길이를 정규화하기 위한 기준 척추 길이 계산
    SPINE_SHOULDER_JOINT = 20
    SPINE_BASE_JOINT = 0
    ref_joint1 = coords[:, :, SPINE_SHOULDER_JOINT:SPINE_SHOULDER_JOINT+1, :]
    ref_joint2 = coords[:, :, SPINE_BASE_JOINT:SPINE_BASE_JOINT+1, :]
    # (T, 2, 1, 1)
    torso_lengths = np.linalg.norm(ref_joint1 - ref_joint2, axis=-1, keepdims=True) + 1e-8
    
    # >> (T, 2, 25, 1) 크기의 0 벡터 초기화
    bone_length_features = np.zeros((T, 2, BASE_NUM_JOINTS, 1))

    for parent, child in SKELETON_BONES:
        # >> 뼈 벡터 계산 (T, 2, 3)
        bone_vec = coords[:, :, child, :] - coords[:, :, parent, :]
        # >> 뼈 길이 계산 (T, 2, 1)
        bone_len = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
        # >> 정규화된 뼈 길이 (T, 2, 1)
        norm_len = bone_len / torso_lengths.squeeze(-1)
        # >> 자식(child) 관절의 특징 채널에 정규화된 뼈 길이 값을 할당
        bone_length_features[:, :, child, 0] = norm_len.squeeze(-1)


    # >> 2-2. 관절 각도 특징 (1D)
    # >> (T, 2, 25, 1) 크기의 0 벡터 초기화
    joint_angle_features = np.zeros((T, 2, BASE_NUM_JOINTS, 1))
    
    for p_idx, j_idx, c_idx in JOINT_ANGLE_TRIPLETS:
        # >> 두 개의 뼈 벡터 계산 (T, 2, 3)
        vec1 = coords[:, :, p_idx, :] - coords[:, :, j_idx, :]
        vec2 = coords[:, :, c_idx, :] - coords[:, :, j_idx, :]

        # >> 벡터 크기 계산 (T, 2)
        mag1 = np.linalg.norm(vec1, axis=-1)
        mag2 = np.linalg.norm(vec2, axis=-1)
        
        # >> 코사인 각도 계산 (T, 2)
        dot_prod = np.einsum('ntc,ntc->nt', vec1, vec2)
        cos_theta = dot_prod / (mag1 * mag2 + 1e-8)
        
        # >> 클리핑 후 아크코사인 변환 (라디안 값) (T, 2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # >> 중심(joint) 관절의 특징 채널에 각도 값을 할당
        joint_angle_features[:, :, j_idx, 0] = angle
        

    # --- 3. [신규] 모든 특징 결합 (7D + 1D + 1D = 9D) ---
    combined_features_per_person = np.concatenate(
        (dynamic_features, bone_length_features, joint_angle_features), 
        axis=-1
    ) # shape: (T, 2, 25, 9)

    
    # >> 두 사람의 데이터를 분리한다. (T, 25, 9)
    person1_features = combined_features_per_person[:, 0, :, :]
    person2_features = combined_features_per_person[:, 1, :, :]
    
    # >> 두 사람의 특징을 관절 축(axis=1)을 따라 연결하여 (T, 50, 9) 형태로 만든다.
    final_features = np.concatenate((person1_features, person2_features), axis=1)
    
    return final_features

# 'calculate_and_save_stats'를 위한 일꾼(worker) 함수
def process_file_for_stats(filename):
    """파일 하나를 받아 통계 계산에 필요한 특징(feature)을 반환합니다."""
    if not filename.endswith('.skeleton'):
        return None  # .skeleton 파일이 아니면 None 반환

    subject_id = int(filename[9:12])
    if subject_id not in TRAINING_SUBJECTS:
        return None  # 훈련용 데이터가 아니면 None 반환

    skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(skeleton_path)
    if coords.shape[0] == 0:
        return None  # 파일 내용이 비어있으면 None 반환

    downsampled_coords = coords[::2, :, :, :]
    
    # >> [수정됨] 9차원 특징을 반환
    features = _calculate_features(downsampled_coords) 
    
    valid_frames = features.reshape(-1, features.shape[-1])
    valid_frames = valid_frames[np.abs(valid_frames).sum(axis=1) > 1e-6]
    
    if valid_frames.shape[0] > 0:
        return valid_frames # 계산된 특징을 반환
    else:
        return None


# ## ------------------------------------------------------------------------------
# 훈련 데이터셋에 대해서만 특징을 계산하고,
# 전체 특징 데이터의 평균과 표준편차를 계산하여 파일로 저장한다.
# 이 통계치는 나중에 모델 훈련 시 데이터 정규화에 사용된다.
# ## ------------------------------------------------------------------------------
def calculate_and_save_stats():
    print("--- 1단계: 훈련 데이터셋 통계치 계산 시작 ---")
    all_features = [] # 모든 훈련 데이터의 특징을 저장할 리스트
    filenames = os.listdir(SOURCE_DATA_PATH)

    # 사용할 CPU 코어 수를 정합니다.
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for stats calculation...")

    all_features = []
    # 멀티프로세싱 Pool을 생성하고 작업을 분배합니다.
    with Pool(processes=num_cores) as pool:
        # pool.imap_unordered를 사용하여 작업을 병렬 처리하고 tqdm으로 진행률 표시
        results_iterator = pool.imap_unordered(process_file_for_stats, filenames)

        # 반환된 결과(None이 아닌 것)를 all_features 리스트에 추가
        for result in tqdm(results_iterator, total=len(filenames), desc="[Calculating Stats]"):
            if result is not None:
                all_features.append(result)

    # >> 모든 특징 데이터를 하나의 큰 numpy 배열로 결합합니다.
    all_features_np = np.concatenate(all_features, axis=0)

    # >> 평균과 표준편차를 계산합니다.
    mean = np.mean(all_features_np, axis=0)
    std_raw = np.std(all_features_np, axis=0)

    epsilon = 1e-6
    std = np.clip(std_raw, a_min=epsilon, a_max=None)

    # >> 계산된 통계치를 .npz 파일로 저장합니다.
    np.savez(STATS_FILE, mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1))
    print(f"통계치 계산 완료. '{STATS_FILE}' 파일에 저장되었습니다.")
    

# 'main' 함수를 위한 일꾼(worker) 함수
def process_and_save_file(filename):
    """파일 하나를 전처리하고 .pt 파일로 저장합니다."""
    if not filename.endswith('.skeleton'):
        return

    skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(skeleton_path)
    
    if coords.shape[0] == 0:
        # >> [수정됨] 9차원 0벡터
        processed_features = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS)) 
        first_frame_coords = np.zeros((NUM_JOINTS, 3))
    else: 
        first_frame_raw = coords[0, :, :, :]
        first_frame_coords = np.concatenate((first_frame_raw[0], first_frame_raw[1]), axis=0)
        
        downsampled_coords = coords[::2, :, :]
        # >> [수정됨] 9차원 특징 계산
        raw_features = _calculate_features(downsampled_coords) 
        
        num_frames = raw_features.shape[0]
        if num_frames < MAX_FRAMES:
            pad_width = MAX_FRAMES - num_frames
            # >> [수정됨] 9차원 0벡터
            padding = np.zeros((pad_width, NUM_JOINTS, config.NUM_COORDS)) 
            processed_features = np.concatenate((raw_features, padding), axis=0)
        else:
            processed_features = raw_features[:MAX_FRAMES]
            
    action_id = int(filename[17:20])
    label = action_id - 1
    
    data_to_save = {
        'data': torch.from_numpy(processed_features).float(),
        'label': torch.tensor(label, dtype=torch.long),
        'first_frame_coords': torch.from_numpy(first_frame_coords).float()
    }
    
    target_filename = filename.replace('.skeleton', '.pt')
    target_filepath = os.path.join(TARGET_DATA_PATH, target_filename)
    torch.save(data_to_save, target_filepath)


# ## ----------------------------------------------------------------------------------
# 함수: main
# 기능: 전체 데이터 전처리 파이프라인을 실행하는 메인 함수입니다.
# 1. 통계치 파일(.npz) 존재 여부 확인 및 생성
# 2. 모든 .skeleton 파일을 순회하며 전처리 수행
# 3. 전처리된 데이터를 .pt 파일로 저장
# ## ----------------------------------------------------------------------------------
def main():
    
    # >> 통계 파일이 없으면 최초 1회 생성한다.
    if not os.path.exists(STATS_FILE):
        calculate_and_save_stats()
        
    print("\n--- 2단계: 특징 계산 및 원본(raw) 데이터 저장 시작 ---")


    # >> 결과물을 저장할 디렉토리가 없다면 생성한다.    
    if not os.path.exists(TARGET_DATA_PATH):
        os.makedirs(TARGET_DATA_PATH)


    filenames = os.listdir(SOURCE_DATA_PATH)

    # 사용할 CPU 코어 수를 정합니다.
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for multiprocessing...")

    # 멀티프로세싱 Pool을 사용하여 파일 저장을 병렬로 처리합니다.
    with Pool(processes=num_cores) as pool:
        # list()로 감싸서 모든 작업이 끝날 때까지 기다립니다.
        list(tqdm(pool.imap_unordered(process_and_save_file, filenames), total=len(filenames), desc="[Saving Raw Features]"))

    print("\n모든 데이터의 원본 특징(raw feature) 저장이 완료되었습니다.")




# >> 이 스크립트가 직접 실행될 때 main() 함수를 호출한다.
if __name__ == '__main__':
    main()
