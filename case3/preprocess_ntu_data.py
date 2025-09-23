# preprocess_ntu_data.py (효율적으로 수정됨)

import os
import numpy as np
import torch
from tqdm import tqdm
import config

SOURCE_DATA_PATH = '../../paper-review/Action_Recognition/Code/nturgbd01/' 
TARGET_DATA_PATH = 'nturgbd_processed/' 
STATS_FILE = 'stats.npz'
MAX_FRAMES = config.MAX_FRAMES
NUM_JOINTS = config.NUM_JOINTS
BASE_NUM_JOINTS = 25
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

def _apply_kalman_filter(coords):
    """
    3D 좌표 시계열 데이터에 칼만 필터를 적용하여 노이즈를 제거하고 움직임을 부드럽게 합니다.
    입력 shape: (num_frames, 2, 25, 3)
    출력 shape: (num_frames, 2, 25, 3)
    """
    num_frames, num_persons, num_joints, num_dims = coords.shape
    filtered_coords = np.zeros_like(coords)

    # 칼만 필터 파라미터 (상수 속도 모델)
    dt = 1.0  # 시간 간격
    F = np.array([[1, dt], [0, 1]])      # 상태 전이 행렬
    H = np.array([[1, 0]])               # 관측 행렬
    Q = np.eye(2) * 0.1                  # 프로세스 노이즈 공분산 (모델의 불확실성)
    R = np.array([[10]])                 # 측정 노이즈 공분산 (센서의 불확실성)

    for p in range(num_persons):
        for j in range(num_joints):
            for d in range(num_dims):
                # 처리할 1D 시계열 데이터
                measurements = coords[:, p, j, d]
                
                # 초기 상태 및 공분산
                x = np.array([[measurements[0]], [0]])  # 초기 상태 [위치, 속도]
                P = np.eye(2)                           # 초기 상태 공분산

                filtered_positions = []

                for z in measurements:
                    # 예측 단계
                    x_pred = F @ x
                    P_pred = F @ P @ F.T + Q

                    # 업데이트 단계
                    y = z - H @ x_pred
                    S = H @ P_pred @ H.T + R
                    K = P_pred @ H.T @ np.linalg.inv(S)
                    x = x_pred + K @ y
                    P = (np.eye(2) - K @ H) @ P_pred
                    
                    filtered_positions.append(x[0, 0])
                
                filtered_coords[:, p, j, d] = filtered_positions
                
    return filtered_coords

def _read_skeleton_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_frames = int(lines[0])
    frame_data = np.zeros((num_frames, 2, BASE_NUM_JOINTS, 3))
    line_idx = 1
    
    
    for i in range(num_frames):
        if line_idx >= len(lines): break

        num_bodies = int(lines[line_idx].strip())
        line_idx += 1

        bodies_read = 0
        for b in range(num_bodies):
            if bodies_read < 2:
                line_idx += 1 # bodyID, clipped, ... 라인 스킵
                num_joints_file = int(lines[line_idx].strip())
                line_idx += 1
                
                joint_coords_frame = np.zeros((num_joints_file, 3))
                for j in range(num_joints_file):
                    joint_info = lines[line_idx].strip().split()
                    joint_coords_frame[j] = [float(coord) for coord in joint_info[:3]]
                    line_idx += 1
                
                frame_data[i, bodies_read, :, :] = joint_coords_frame
                bodies_read += 1
            # 2명을 초과하는 나머지 사람 데이터는 건너뜀
            else:
                line_idx += 1
                other_num_joints = int(lines[line_idx].strip())
                line_idx += (1 + other_num_joints)
                
    return frame_data

def _normalize_by_bone_length(coords):
    """
    척추 길이를 기준으로 3D 스켈레톤 좌표를 정규화합니다.
    입력 shape: (num_frames, 2, 25, 3)
    출력 shape: (num_frames, 2, 25, 3)
    """
    # 기준 관절 인덱스 정의
    SPINE_SHOULDER_JOINT = 20
    SPINE_BASE_JOINT = 0
    
    # 각 프레임, 각 사람(person)에 대해 기준 관절의 좌표를 선택합니다.
    # shape: (num_frames, 2, 1, 3)
    ref_joint1_coords = coords[:, :, SPINE_SHOULDER_JOINT:SPINE_SHOULDER_JOINT+1, :]
    ref_joint2_coords = coords[:, :, SPINE_BASE_JOINT:SPINE_BASE_JOINT+1, :]
    
    # 척추 길이(유클리드 거리)를 계산합니다. 0으로 나누는 것을 방지하기 위해 작은 값(epsilon)을 더합니다.
    # shape: (num_frames, 2, 1)
    torso_lengths = np.linalg.norm(ref_joint1_coords - ref_joint2_coords, axis=-1) + 1e-8

    # 브로드캐스팅을 위해 차원을 추가해줍니다. (num_frames, 2, 1) -> (num_frames, 2, 1, 1)
    torso_lengths = np.expand_dims(torso_lengths, axis=-1)
    
    # 모든 관절 좌표를 해당 프레임의 척추 길이로 나눕니다.
    normalized_coords = coords / torso_lengths
    
    return normalized_coords

def _calculate_features(coords):
    center_joint = coords[:, :, 0:1, :]
    coords = coords - center_joint

    displacement = np.zeros_like(coords)
    displacement[1:] = coords[1:] - coords[:-1]

    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)

    magnitude = np.linalg.norm(displacement, axis=-1, keepdims=True) + 1e-8
    direction = displacement / magnitude

    acceleration = np.zeros_like(displacement)
    acceleration[1:] = displacement[1:] - displacement[:-1]

    combined_features_per_person = np.concatenate((distance, direction, acceleration), axis=-1)

    person1_features = combined_features_per_person[:, 0, :, :]
    person2_features = combined_features_per_person[:, 1, :, :]
    
    final_features = np.concatenate((person1_features, person2_features), axis=1)
    
    return final_features


def calculate_and_save_stats():
    print("--- 1단계: 훈련 데이터셋 통계치 계산 시작 ---")
    all_features = []
    filenames = os.listdir(SOURCE_DATA_PATH)
    process_bar = tqdm(filenames, desc="[Calculating Stats]")
    for filename in process_bar:
        if not filename.endswith('.skeleton'): continue
        subject_id = int(filename[9:12])

        if subject_id not in TRAINING_SUBJECTS: continue

        skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(skeleton_path)

        if coords.shape[0] == 0: continue

        # 1. 칼만 필터 적용
        smoothed_coords = _apply_kalman_filter(coords)
        
        # 2. 뼈 길이 정규화 적용
        normalized_coords = _normalize_by_bone_length(smoothed_coords)

        # 3. 다운샘플링
        downsampled_coords = normalized_coords[::2, :, :]

        # 4. 전처리된 좌표로 특징 계산
        features = _calculate_features(downsampled_coords)
        
        valid_frames = features.reshape(-1, features.shape[-1])
        valid_frames = valid_frames[np.abs(valid_frames).sum(axis=1) > 1e-6]
        if valid_frames.shape[0] > 0:
            all_features.append(valid_frames)

    all_features_np = np.concatenate(all_features, axis=0)
    mean = np.mean(all_features_np, axis=0)
    std_raw = np.std(all_features_np, axis=0)
    epsilon = 1e-6  # 0으로 나누는 것을 방지하기 위한 아주 작은 값
    std = np.clip(std_raw, a_min=epsilon, a_max=None)
    
    np.savez(STATS_FILE, mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1))
    print(f"통계치 계산 완료. '{STATS_FILE}' 파일에 저장되었습니다.")

    
def main():
    """특징을 계산하고 정규화 없이 그대로 저장하는 메인 함수"""
    
    # 통계 파일이 없으면 최초 1회 생성
    if not os.path.exists(STATS_FILE):
        calculate_and_save_stats()
        
    print("\n--- 2단계: 특징 계산 및 원본(raw) 데이터 저장 시작 ---")
    
    if not os.path.exists(TARGET_DATA_PATH):
        os.makedirs(TARGET_DATA_PATH)

    filenames = os.listdir(SOURCE_DATA_PATH)
    process_bar = tqdm(filenames, desc="[Saving Raw Features]")
    for filename in process_bar:
        if not filename.endswith('.skeleton'): continue
        
        skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(skeleton_path)
        
        if coords.shape[0] == 0:
            processed_features = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS))
            first_frame_coords = np.zeros((NUM_JOINTS, 3))
        else:
            first_frame_raw = coords[0, :, :, :]
            first_frame_coords = np.concatenate((first_frame_raw[0], first_frame_raw[1]), axis=0)

            # 칼만 필터 적용
            smoothed_coords = _apply_kalman_filter(coords)

            # 뼈길이 정규화
            normalized_coords = _normalize_by_bone_length(smoothed_coords)

            # 다운샘플링
            normalized_coords = normalized_coords[::2, :, :, :]

            # 특징 계산
            raw_features = _calculate_features(normalized_coords)
            
            
            num_frames = raw_features.shape[0]
            if num_frames < MAX_FRAMES:
                pad_width = MAX_FRAMES - num_frames
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

    print("\n모든 데이터의 원본 특징(raw feature) 저장이 완료되었습니다.")

if __name__ == '__main__':
    main()
