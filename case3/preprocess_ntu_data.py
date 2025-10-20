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
import cv2
from multiprocessing import Pool, cpu_count

# >> 처리할 NTU_RGB+D 60 skeleton 데이터가 있는 위치
SOURCE_DATA_PATH = '../../paper-review/Action_Recognition/Code/nturgbd01/' 
# >> 처리가 완료된 파일들을 저장할 위치
TARGET_DATA_PATH = '../nturgbd_processed/'
# >> 훈련 데이터셋의 평균과 표준편차를 저장할 파일 이름 
STATS_FILE = '../stats.npz'
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

# 각 관절의 3D 좌표 시계열 데이터에 칼만 필터를 적용하여 노이즈를 제거하고 스무딩합니다.
# 최적화를 위해 각 관절(2명 * 25관절 = 50개)마다 독립적인 칼만 필터를 생성하여 관리합니다.
# ## --------------------------------------------------------------------------------
def _apply_kalman_filter(coords):
    num_frames, num_persons, num_joints, _ = coords.shape
    smoothed_coords = np.zeros_like(coords)

    for p in range(num_persons):
        for j in range(num_joints):
            # >> 각 관절의 시계열 데이터를 추출합니다. (num_frames, 3)
            joint_series = coords[:, p, j, :]

            # 파일의 첫 프레임이 비어있는 경우(모두 0) 필터링을 건너뜁니다.
            if np.all(joint_series[0] == 0):
                smoothed_coords[:, p, j, :] = joint_series
                continue

            kalman = cv2.KalmanFilter(6, 3)
            kalman.transitionMatrix = np.array([
                [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
            kalman.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]
            ], dtype=np.float32)
            kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
            kalman.processNoiseCov[3:, 3:] *= 10
            kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
            
            # >> reshape(3, 1)을 제거하여 1차원 배열로 할당합니다.
            kalman.statePost = np.zeros(6, dtype=np.float32)
            kalman.statePost[:3] = joint_series[0] # 첫 프레임 좌표로 초기 위치 설정

            for t in range(num_frames):
                prediction = kalman.predict()
                measurement = joint_series[t].astype(np.float32)
                
                if np.all(measurement == 0):
                    corrected_state = prediction
                else:
                    # >> measurement를 (3, 1) 형태의 열벡터로 변환하여 전달합니다.
                    corrected_state = kalman.correct(measurement.reshape(3, 1))
                
                smoothed_coords[t, p, j, :] = corrected_state[:3].flatten()

    return smoothed_coords



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
# 좌표로부터 이동 거리, 방향, 가속도 등의 특징을 계산한다.
# shape: (num_frames, 50, 7)
# (50 = 25관절 * 2명, 7 = 거리(1) + 방향(3) + 가속도(3))
# ## ------------------------------------------------------------------------------
def _calculate_features(coords):
    # >> 스켈레톤 중심화
    # >> 골반 중심(0번 관절)을 원점(0,0,0)으로 만들어 전체 스켈레톤의 위치 변화에 무관하도록 한다.
    center_joint = coords[:, :, 0:1, :]
    coords = coords - center_joint


    # >> 변위 계산
    # >> 현재 프레임과 이전 프레임의 좌표 차이를 계산하여 각 관절의 움직임을 나타낸다.
    displacement = np.zeros_like(coords)
    displacement[1:] = coords[1:] - coords[:-1] # 첫 프레임(인덱스 0)의 변위는 0이므로, 두 번째 프레임부터 계산


    # >> 거리 계산
    # >> 변위 벡터의 크기(L2 norm)를 계산하여 스칼라 값인 이동 거리를 구한다.
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)


    # >> 방향 계산
    # >> 변위 벡터를 자신의 크기로 나누어 단위 벡터(방향)를 구한다.
    magnitude = np.linalg.norm(displacement, axis=-1, keepdims=True) + 1e-8
    direction = displacement / magnitude


    # >> 가속도 계산
    # >> 변위의 변화율, 즉 현재 변위와 이전 변위의 차이를 계산한다.
    acceleration = np.zeros_like(displacement)
    acceleration[1:] = displacement[1:] - displacement[:-1]


    # >> 특징 결합
    # >> 계산된 거리(1차원), 방향(3차원), 가속도(3차원)를 합쳐 7차원의 특징 벡터를 만든다.
    combined_features_per_person = np.concatenate((distance, direction, acceleration), axis=-1)


    # >> 두 사람의 데이터를 분리한다.
    person1_features = combined_features_per_person[:, 0, :, :]
    person2_features = combined_features_per_person[:, 1, :, :]
    

    # >> 두 사람의 특징을 관절 축(axis=1)을 따라 연결하여 (num_frames, 50, 7) 형태로 만든다.
    final_features = np.concatenate((person1_features, person2_features), axis=1)
    
    return final_features

def process_file_for_stats(filename):
    """통계 계산을 위해 단일 파일을 처리하고 특징(feature)을 반환합니다."""
    if not filename.endswith('.skeleton'): return None
    
    subject_id = int(filename[9:12])
    if subject_id not in TRAINING_SUBJECTS: return None

    skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(skeleton_path)
    if coords.shape[0] == 0: return None

    smoothed_coords = _apply_kalman_filter(coords)
    downsampled_coords = smoothed_coords[::2, :, :, :]
    features = _calculate_features(downsampled_coords)
    
    valid_frames = features.reshape(-1, features.shape[-1])
    valid_frames = valid_frames[np.abs(valid_frames).sum(axis=1) > 1e-6]
    
    if valid_frames.shape[0] > 0:
        return valid_frames
    return None


# ## ------------------------------------------------------------------------------
# 훈련 데이터셋에 대해서만 특징을 계산하고,
# 전체 특징 데이터의 평균과 표준편차를 계산하여 파일로 저장한다.
# 이 통계치는 나중에 모델 훈련 시 데이터 정규화에 사용된다.
# ## ------------------------------------------------------------------------------
def calculate_and_save_stats():
    print("--- 1단계: 훈련 데이터셋 통계치 계산 시작 ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    
    # 사용할 CPU 코어 수를 정합니다.
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for stats calculation...")

    all_features = []
    # 멀티프로세싱 Pool을 사용하여 병렬 처리
    with Pool(processes=num_cores) as pool:
        # pool.imap_unordered를 사용하여 결과를 비동기적으로 받아옵니다.
        results = list(tqdm(pool.imap_unordered(process_file_for_stats, filenames), total=len(filenames), desc="[Calculating Stats]"))

    # 유효한 결과(None이 아닌 것)만 리스트에 추가합니다.
    for result in results:
        if result is not None:
            all_features.append(result)

    all_features_np = np.concatenate(all_features, axis=0)
    mean = np.mean(all_features_np, axis=0)
    std_raw = np.std(all_features_np, axis=0)
    epsilon = 1e-6
    std = np.clip(std_raw, a_min=epsilon, a_max=None)
    
    np.savez(STATS_FILE, mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1))
    print(f"통계치 계산 완료. '{STATS_FILE}' 파일에 저장되었습니다.")


def process_file(filename):
    """하나의 스켈레톤 파일을 읽고, 필터링하고, 특징을 계산하여 저장합니다."""
    if not filename.endswith('.skeleton'):
        return  # .skeleton 파일이 아니면 아무것도 하지 않음

    skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(skeleton_path)

    if coords.shape[0] == 0:
        processed_features = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS))
        first_frame_coords = np.zeros((NUM_JOINTS, 3))
    else:
        smoothed_coords = _apply_kalman_filter(coords)
        first_frame_raw = smoothed_coords[0, :, :, :]
        first_frame_coords = np.concatenate((first_frame_raw[0], first_frame_raw[1]), axis=0)
        
        downsampled_coords = smoothed_coords[::2, :, :]
        raw_features = _calculate_features(downsampled_coords)

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

    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for multiprocessing...")
    
    # >> Pool 객체를 생성하고, map 함수로 작업을 분배합니다.
    with Pool(processes=num_cores) as pool:
        # >> tqdm을 멀티프로세싱에 적용하여 진행 상황을 봅니다.
        list(tqdm(pool.imap_unordered(process_file, filenames), total=len(filenames), desc="[Saving Raw Features]"))


    print("\n모든 데이터의 원본 특징(raw feature) 저장이 완료되었습니다.")




# >> 이 스크립트가 직접 실행될 때 main() 함수를 호출한다.
if __name__ == '__main__':
    main()
