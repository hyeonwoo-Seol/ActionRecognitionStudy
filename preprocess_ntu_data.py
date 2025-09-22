# preprocess_ntu_data.py (효율적으로 수정됨)

import os
import numpy as np
import torch
from tqdm import tqdm

# --- 설정 (이전과 동일) ---
SOURCE_DATA_PATH = '../paper-review/Action_Recognition/Code/nturgbd01/' 
TARGET_DATA_PATH = 'nturgbd_processed/' 
STATS_FILE = 'stats.npz'
MAX_FRAMES = 150
NUM_JOINTS = 25
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

def _read_skeleton_file(file_path):
    # (이전과 동일)
    # ... (생략) ...
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_frames = int(lines[0])
    frame_data = []
    line_idx = 1
    for _ in range(num_frames):
        if line_idx >= len(lines): break
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1
        if num_bodies > 0:
            line_idx += 1 
            num_joints_file = int(lines[line_idx].strip())
            line_idx += 1
            joint_coords_frame = np.zeros((num_joints_file, 3))
            for j in range(num_joints_file):
                joint_info = lines[line_idx].strip().split()
                joint_coords_frame[j] = [float(coord) for coord in joint_info[:3]]
                line_idx += 1
            frame_data.append(joint_coords_frame)
            for _ in range(num_bodies - 1):
                line_idx += 1
                other_num_joints = int(lines[line_idx].strip())
                line_idx += (1 + other_num_joints)
        else:
            frame_data.append(np.zeros((NUM_JOINTS, 3)))
    return np.array(frame_data)


def _calculate_features(coords):
    # (이전과 동일)
    center_joint = coords[:, 0:1, :]
    coords = coords - center_joint
    displacement = np.zeros_like(coords)
    displacement[1:] = coords[1:] - coords[:-1]
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)
    displacement_xy = displacement[..., :2] 
    magnitude_xy = np.linalg.norm(displacement_xy, axis=-1, keepdims=True) + 1e-8
    direction_xy = displacement_xy / magnitude_xy
    acceleration = np.zeros_like(displacement)
    acceleration[1:] = displacement[1:] - displacement[:-1]
    combined_features = np.concatenate((distance, direction_xy, acceleration), axis=-1)
    return combined_features

def calculate_and_save_stats():
    # (이전과 동일, 최초 1회 통계 계산을 위해 필요)
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
        coords = coords[::2, :, :]
        features = _calculate_features(coords)
        valid_frames = features.reshape(-1, features.shape[-1])
        valid_frames = valid_frames[np.abs(valid_frames).sum(axis=1) > 1e-6]
        if valid_frames.shape[0] > 0:
            all_features.append(valid_frames)
    all_features_np = np.concatenate(all_features, axis=0)
    mean = np.mean(all_features_np, axis=0)
    std = np.std(all_features_np, axis=0)
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
            processed_features = np.zeros((MAX_FRAMES, NUM_JOINTS, 6))
        else:
            coords = coords[::2, :, :]
            # ## <<-- 변경점: 정규화 과정 삭제 -->> ##
            raw_features = _calculate_features(coords)
            
            # 패딩/절단
            num_frames = raw_features.shape[0]
            if num_frames < MAX_FRAMES:
                pad_width = MAX_FRAMES - num_frames
                padding = np.zeros((pad_width, NUM_JOINTS, 6))
                processed_features = np.concatenate((raw_features, padding), axis=0)
            else:
                processed_features = raw_features[:MAX_FRAMES]
        
        action_id = int(filename[17:20])
        label = action_id - 1

        data_to_save = {
            'data': torch.from_numpy(processed_features).float(),
            'label': torch.tensor(label, dtype=torch.long)
        }

        target_filename = filename.replace('.skeleton', '.pt')
        target_filepath = os.path.join(TARGET_DATA_PATH, target_filename)
        torch.save(data_to_save, target_filepath)

    print("\n모든 데이터의 원본 특징(raw feature) 저장이 완료되었습니다.")

if __name__ == '__main__':
    main()
