# preprocess_ntu_data.py
# ##--------------------------------------------------------------------------
# x, y, z skeleton 좌표를 이동 거리, 방향, 가속도로 미리 변환하는 파일이다.
# 구현해야 할 기능
# 1. 파일 읽어오기
# 2. 좌표값을 내가 원하는 형태로 변환하기
# 3. 뼈 길이 정규화가 성능에 영향을 주는지 확인
# 4. 평균과 표준편차를 계산하기
# 5. 이를 통합하는 main 함수 만들기
# ##----------------------------------------------------------------------------

import os
import numpy as np
import torch
from tqdm import tqdm
import config




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
# 정규화된 좌표로부터 이동 거리, 방향, 가속도 등의 특징을 계산한다.
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




# ## ------------------------------------------------------------------------------
# 훈련 데이터셋에 대해서만 특징을 계산하고,
# 전체 특징 데이터의 평균과 표준편차를 계산하여 파일로 저장한다.
# 이 통계치는 나중에 모델 훈련 시 데이터 정규화에 사용된다.
# ## ------------------------------------------------------------------------------
def calculate_and_save_stats():
    print("--- 1단계: 훈련 데이터셋 통계치 계산 시작 ---")
    all_features = [] # 모든 훈련 데이터의 특징을 저장할 리스트
    filenames = os.listdir(SOURCE_DATA_PATH)


    process_bar = tqdm(filenames, desc="[Calculating Stats]")
    for filename in process_bar:
        if not filename.endswith('.skeleton'): continue # .skeleton 파일이 아니면 건너뛴다.
        
        
        # >> # 파일 이름에서 피실험자 ID를 추출한다.
        subject_id = int(filename[9:12])

        if subject_id not in TRAINING_SUBJECTS: continue # 피실험자 ID가 훈련용 ID 목록에 없으면 건너뛴다.


        # >> 파일 읽기
        skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(skeleton_path)

        if coords.shape[0] == 0: continue # 파일 내용이 비어있으면 건너뛴다.


        # >> 뼈 길이 정규화
        # >> 실험을 위해 사용하지 않는다.
        # normalized_coords = _normalize_by_bone_length(coords)

        # >> 다운샘플링
        # downsampled_coords = normalized_coords[::2, :, :]
        downsampled_coords = coords[::2, :, :, :]
        
        # >> 전처리된 좌표로 특징 계산
        features = _calculate_features(downsampled_coords)
        

        # >> 모든 프레임과 관절을 하나의 차원으로 펼친다.
        # >> (num_frames * 50, 7)
        valid_frames = features.reshape(-1, features.shape[-1])


        # >> 특징 값이 거의 0인, 즉 움직임이 없는 프레임을 제외한다.
        valid_frames = valid_frames[np.abs(valid_frames).sum(axis=1) > 1e-6]
        if valid_frames.shape[0] > 0:
            all_features.append(valid_frames)


    # >> 모든 특징 데이터를 하나의 큰 numpy 배열로 결합한다.
    all_features_np = np.concatenate(all_features, axis=0)


    # >> 평균과 표준편차를 계산한다.
    mean = np.mean(all_features_np, axis=0)
    std_raw = np.std(all_features_np, axis=0)
    

    epsilon = 1e-6  # 0으로 나누는 것을 방지하기 위한 아주 작은 값
    std = np.clip(std_raw, a_min=epsilon, a_max=None)
    

    # >> 계산된 통계치를 .npz 파일로 저장합니다. 나중에 불러오기 쉽도록 reshape 한다.
    np.savez(STATS_FILE, mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1))
    print(f"통계치 계산 완료. '{STATS_FILE}' 파일에 저장되었습니다.")




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
    process_bar = tqdm(filenames, desc="[Saving Raw Features]")
    for filename in process_bar:
        if not filename.endswith('.skeleton'): continue # .skeleton 파일이 아니라면 해당 파일은 넘어간다.
        
        # >> 디렉터리 경로와 파일 이름을 결합한다.
        # >> os.path.join이 운영체제에 적합한 경로 구분자를 알아서 사용해준다.
        skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
        
        # >> 위에서 결합한 경로에서 .skeleton 파일을 읽어서 안에 있는 x, y, z 좌표를 담는다.
        coords = _read_skeleton_file(skeleton_path)
        

        # >> 파일이 비어있는 경우, 0으로 채워진 더미 데이터를 생성한다.
        if coords.shape[0] == 0:
            processed_features = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS))
            first_frame_coords = np.zeros((NUM_JOINTS, 3))
        else: 
            # >> 시각화 등에 사용할 첫 번째 프레임의 원본 좌표를 저장한다.
            first_frame_raw = coords[0, :, :, :]
            first_frame_coords = np.concatenate((first_frame_raw[0], first_frame_raw[1]), axis=0)


            # >> 데이터를 전처리한다.
            # >> 뼈길이 정규화
            # >> 실험을 위해 사용하지 않는다.
            # normalized_coords = _normalize_by_bone_length(coords)

            # >> 다운샘플링
            # normalized_coords = normalized_coords[::2, :, :, :] # 뼈 정규화를 사용하는 코드 
            downsampled_coords = coords[::2, :, :]
            
            # >> 특징 계산
            # raw_features = _calculate_features(normalized_coords)  # 뼈 정규화를 사용하는 코드 
            raw_features = _calculate_features(downsampled_coords)

            # >> 패딩 및 자르기 (Padding & Truncating)
            # 모든 데이터의 프레임 길이를 MAX_FRAMES로 통일한다.
            num_frames = raw_features.shape[0]
            if num_frames < MAX_FRAMES: # 프레임 수가 MAX_FRAMES보다 작으면, 부족한 만큼 0으로 채운다.
                pad_width = MAX_FRAMES - num_frames
                padding = np.zeros((pad_width, NUM_JOINTS, config.NUM_COORDS))
                processed_features = np.concatenate((raw_features, padding), axis=0)
            else: # 프레임 수가 MAX_FRAMES보다 크면, MAX_FRAMES까지만 사용한다.
                processed_features = raw_features[:MAX_FRAMES]


        # >> 파일 이름에서 액션 ID를 추출하고, 0부터 시작하는 레이블로 변환한다. (A001 -> 0)
        action_id = int(filename[17:20])
        label = action_id - 1


        # >> 저장할 데이터를 딕셔너리 형태로 구성한다.
        data_to_save = {
            'data': torch.from_numpy(processed_features).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'first_frame_coords': torch.from_numpy(first_frame_coords).float()
        }


        # >> 저장할 파일 경로를 설정하고, .pt 파일을 해당 경로에 저장한다.
        target_filename = filename.replace('.skeleton', '.pt')
        target_filepath = os.path.join(TARGET_DATA_PATH, target_filename)
        torch.save(data_to_save, target_filepath)

    print("\n모든 데이터의 원본 특징(raw feature) 저장이 완료되었습니다.")




# >> 이 스크립트가 직접 실행될 때 main() 함수를 호출한다.
if __name__ == '__main__':
    main()
