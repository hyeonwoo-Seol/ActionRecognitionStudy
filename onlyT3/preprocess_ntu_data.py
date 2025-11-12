# preprocess_ntu_data.py
# ##--------------------------------------------------------------------------
# x, y, z 스켈레톤 좌표를 15가지의 복합 특징(거리, 방향, 뼈 길이, 
# 각종 각도 및 관절 간 거리 등)으로 미리 변환하는 파일이다.
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
TARGET_DATA_PATH = '../nturgbd_processed_allNew200/'
# >> 훈련 데이터셋의 평균과 표준편차를 저장할 파일 이름 
STATS_FILE = '../stats_allNew200.npz'
# >> 모델에 입력으로 사용할 최대 프레임 수
# >> 이보다 길면 잘라내고, 짧으면 0으로 채운다.
MAX_FRAMES = config.MAX_FRAMES
# >> 모델에서 사용할 총 관절 수
# >> NTU 데이터는 최대 2명의 사람을 포함하므로, 25 * 2 = 50이 된다.
NUM_JOINTS = config.NUM_JOINTS
# >> NTU 데이터셋의 기본 관절 수
BASE_NUM_JOINTS = 25
# >> 파일에서 임시로 읽어들일 최대 사람 수
MAX_BODIES = 5
# >> 훈련에 사용할 피실험자(subject) ID 목록
# >> 이 ID를 기준으로 훈련 데이터와 검증/테스트 데이터를 나눈다.
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

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

# ## ---------------------------------------------------------------------------------
# 관절 각도 계산을 위한 (조부모, 부모, 자식) 관절 모음이다.
# 하나의 각도를 정의하려면 3개의 관절이 필요하다. 
# 부모 관절은 꼭짓점 역할을 하고, 조부모 관절은 꼭짓점과 연결되는 첫 번째 뼈 벡터의 끝점이고
# 자식 관절은 꼭짓점과 연결된 두 번째 뼈 벡터의 끝점이다.
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
    # >> 첫 번째 라인에서 전체 프레임 수를 먼저 읽는다.
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            num_frames = int(first_line)
    except (FileNotFoundError, ValueError, IOError):
        print(f"Error: Could not read frame count from {filepath}")
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    # >> 파일의 첫 줄에서 읽어온 프레임 수가 0인 경우,
    # >> (0, 2, BASE_NUM_JOINTS, 3) 크기의 배열을 반환한다.
    if num_frames == 0:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    # >> 1. 첫 번째 스캔: 실제 데이터가 있는 bodyID 2개 찾기
    # >> 파일 전체를 미리 스캔하여 두 명의 사람이 누구인지 미리 결정한다.
    # >> NTU에서는 한 프레임에 2명 이상의 스켈레톤을 포함할 수 있기 때문이다.
    # >> 이 때 투표 방식을 사용한다.
    body_id_counts = {} # 투표함 준비
    try:
        with open(filepath, 'r') as f:
            f.readline()  # 첫 번째 프레임 수 라인 건너뛰기

            line_idx = 1
            while True: # 파일 전체를 한 프레임씩, 한 사람씩 스캔하기.
                line = f.readline()
                if not line: break # 파일 끝
                line_idx += 1

                # >> 파일을 읽다가 데이터가 손상된 프레임을 만나도 프로그램을 멈추지 말고 경고만 출력한다.
                try:
                    # >> 파일에서 읽어온 한 줄의 양쪽 공백이나 줄바꿈 문자를 제거한다.
                    num_bodies = int(line.strip())
                except ValueError: # 파일이 손상되면,,
                    print(f"Warning: Invalid body count at line {line_idx} in {filepath}. Skipping frame.")
                    continue

                # >> 사람 수 만큼 반복한다.
                for i in range(num_bodies):
                    line = f.readline() # body info line을 읽는다.
                    if not line: break
                    line_idx += 1
                    
                    body_info = line.strip().split() # 읽어온 줄을 공백 기준으로 쪼개서 리스트로 만든다.
                    if len(body_info) < 1: # 파일이 손상되면,,
                        print(f"Warning: Skipping empty body info line {line_idx} in {filepath}")
                        continue
                    
                    body_id = body_info[0] # 리스트의 첫 번째 요소가 사람의 고유 ID이므로 이를 가져온다.

                    line = f.readline() # 관절 수가 적힌 다음줄을 읽는다.
                    if not line: break
                    line_idx += 1

                    # >> 읽어온 관절 수를 정수로 변환한다.
                    try:
                        num_joints = int(line.strip())
                    except ValueError:
                        print(f"Warning: Invalid joint count at line {line_idx} in {filepath}.")
                        num_joints = 0
                    
                    
                    # >> 이 body가 유효한지(좌표가 0이 아닌지) 확인한다.
                    has_non_zero_coord = False
                    for j in range(num_joints):
                        line = f.readline()
                        if not line: break
                        line_idx += 1
                        
                        if not has_non_zero_coord:
                            try:
                                joint_info = line.strip().split()
                                # >> 과절의 x, y, z 좌표 중 하나라도 0이 아닌 값이 있는지 확인
                                # >> 모든 관절의 모든 좌표가 0이면 가짜 데이터이다.
                                if any(float(coord) != 0.0 for coord in joint_info[:3]):
                                    has_non_zero_coord = True
                            except (ValueError, IndexError):
                                continue
                    
                    # >> 유효한 데이터를 가진 body_id만 투표
                    if has_non_zero_coord:
                        body_id_counts[body_id] = body_id_counts.get(body_id, 0) + 1
    except IOError as e:
        print(f"Error during Pass 1 scan of {filepath}: {e}")
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))


    # >> 카운트를 기준으로 ID 정렬 (등장 횟수가 많은 순)
    sorted_body_ids = sorted(body_id_counts.items(), key=lambda item: item[1], reverse=True)
    
    # >> 투표 수를 보고 핵심이 되는 두 사람을 확정한다.
    body1_id = sorted_body_ids[0][0] if len(sorted_body_ids) > 0 else None
    body2_id = sorted_body_ids[1][0] if len(sorted_body_ids) > 1 else None

    # >> 2. 두 번째 스캔: 결정된 ID를 기준으로 3D 좌표를 추출해서 넘파이 배열을 만든다.
    final_coords = np.zeros((num_frames, 2, BASE_NUM_JOINTS, 3)) # 모든 데이터가 0으로 채워진 배열
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
                    continue # 1차 스캔에서 이미 경고함

                # >> 총 사람 수 만큼 반복한다.
                for i in range(num_bodies):
                    line = f.readline()
                    if not line: break
                    line_idx += 1
                    
                    body_info = line.strip().split() # 읽어온 줄을 공백 기준으로 쪼개서 리스트로 만든다.
                    if len(body_info) < 1: continue

                    current_body_id = body_info[0] # 지금 읽고 있는 사람의 ID를 저장한다.
                    
                    line = f.readline()
                    if not line: break
                    line_idx += 1
                    
                    try:
                        num_joints = int(line.strip())
                    except ValueError:
                        num_joints = 0

                    target_person_idx = -1
                    # >> current_body_id가 1차 스캔에서 찾은 ID와 일치하는지 확인한다.
                    if current_body_id == body1_id:
                        target_person_idx = 0
                    elif current_body_id == body2_id:
                        target_person_idx = 1
                    
                    # >> 총 관절 수 만큼 반복한다.
                    for j in range(num_joints):
                        line = f.readline() # joint info line
                        if not line: break
                        line_idx += 1
                        
                        # >> target_person_idx가 0 또는 1로 선택된 사람일 경우에만 좌표 값을 읽어서
                        # >> final_coords에 저장한다. 만약 3번째 이상의 사람이라면 저장하지 않는다.
                        if target_person_idx != -1 and j < BASE_NUM_JOINTS:
                            try:
                                joint_info = line.strip().split()
                                x, y, z = map(float, joint_info[:3])
                                final_coords[f_idx, target_person_idx, j] = [x, y, z]
                            except (ValueError, IndexError):
                                continue
                f_idx += 1
    except IOError as e:
        print(f"Error during Pass 2 scan of {filepath}: {e}")
        # >> 데이터가 일부만 채워졌을 수 있으므로 빈 배열 반환한다.
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    return final_coords





# ## ------------------------------------------------------------------------------
# 두 3D 벡터 배치 간의 각도를 계산하는 헬퍼 함수이다.
# vec1, vec2 shape: (T, 2, 3) -> angle shape: (T, 2)
# ## ------------------------------------------------------------------------------
def _calculate_angle_between_vectors(vec1, vec2):
    # >> 단위 벡터로 정규화한다.
    vec1_norm = vec1 / (np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-8)
    
    # >> 내적(dot product) 계산한다.
    # (T, 2, 3) * (T, 2, 3) -> (T, 2)
    dot_prod = np.einsum('ntc,ntc->nt', vec1_norm, vec2_norm)
    
    # >> 클리핑 후 아크코사인. 라디안 값이다.
    angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))
    return angle




# ## ------------------------------------------------------------------------------
# 좌표로부터 동적 특징(4D)과 정적/상호작용 특징(11D)을 계산한다.
# shape: (num_frames, 50, 15)
# 50 = 25관절 * 2명, 15 = 거리(1) + 방향(3) + 뼈길이(1) + 관절각도(1) + 
# 몸통상대각도(2) + 비인접관절거리(2) + P0-P1중심거리(1) + P0-P1손발거리(4)
# ## ------------------------------------------------------------------------------
def _calculate_features(coords):
    # >> shape: (T, 2, 25, 3)
    T = coords.shape[0]
    if T == 0:
        # >> 빈 프레임이면 (0, 50, 9) 형태의 빈 배열을 반환한다.
        return np.zeros((0, NUM_JOINTS, config.NUM_COORDS))

    # >> 뼈 길이를 정규화하기 위한 기준 척추 길이 계산
    SPINE_SHOULDER_JOINT = 20 # 인덱스 20 어깨 중심
    SPINE_BASE_JOINT = 0 # 인덱스 0 골반 중심
    ref_joint1 = coords[:, :, SPINE_SHOULDER_JOINT:SPINE_SHOULDER_JOINT+1, :] # 20번 좌표 가져온다.
    ref_joint2 = coords[:, :, SPINE_BASE_JOINT:SPINE_BASE_JOINT+1, :] # 0번 좌표 가져온다.
    
    # >> +1e-8을 제거하고 순수한 길이를 계산한다.
    # >> 데이터가 진짜인지 가짜인지 확인하기 위함이다.
    # >> shape: (T, 2, 1, 1)
    torso_lengths = np.linalg.norm(ref_joint1 - ref_joint2, axis=-1, keepdims=True)
    
    # >> 유효성 마스크를 생성한다.
    # >> 위에서 생성한 순수 길이를 사용해 불량 데이터를 필터링한다.
    # >> 몸통 길이가 최소 1cm (0.01m) 이상인 프레임/사람만 유효하다고 간주한다.
    MIN_TORSO_LENGTH = 0.01 
    valid_mask = (torso_lengths > MIN_TORSO_LENGTH).astype(np.float32)

    # >> 0으로 나누는 것을 방지하기 위한 '안전한' 버전의 척추 길이이다.
    # >> 나중에 마스킹되어 0으로 처리되기 때문에 1e-8 대신 1.0을 더해도 되지만, 
    # >> 일관성을 위해 1e-8을 더한다.
    safe_torso_lengths = torso_lengths + 1e-8 
    safe_torso_lengths_squeezed = safe_torso_lengths.squeeze() # (T, 2)


    # >> 1. 동적 특징 계산 (4D)
    # >> 원본 coords (T, 2, 25, 3)을 기준으로 변위 계산 
    displacement = np.zeros_like(coords)
    displacement[1:] = coords[1:] - coords[:-1]
    
    # >> 거리 계산
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)

    # >> 방향 계산
    magnitude_disp = np.linalg.norm(displacement, axis=-1, keepdims=True) + 1e-8
    direction = displacement / magnitude_disp

    # >> 동적 특징 결합 (T, 2, 25, 4)
    dynamic_features = np.concatenate((distance, direction), axis=-1)


    # >> 2. 정적 특징 계산 (2D)    
    # >> 2-1. 뼈 길이 특징 (1D)
    # >> (T, 2, 25, 1) 크기의 0 벡터를 초기화한다.
    bone_length_features = np.zeros((T, 2, BASE_NUM_JOINTS, 1))

    for parent, child in SKELETON_BONES:
        # >> 뼈 벡터 계산 (T, 2, 3)
        bone_vec = coords[:, :, child, :] - coords[:, :, parent, :]
        # >> 뼈 길이 계산 (T, 2, 1)
        bone_len = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
        
        # >> 0으로 나누는 것을 방지하는 '안전한' 척추 길이로 나눈다.
        norm_len = bone_len / safe_torso_lengths.squeeze(-1)
        
        # >> 자식(child) 관절의 특징 채널에 정규화된 뼈 길이 값을 할당한다.
        bone_length_features[:, :, child, 0] = norm_len.squeeze(-1)


    # >> 2-2. 관절 각도 특징 (1D)
    # >> (T, 2, 25, 1) 크기의 0 벡터를 초기화한다.
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
        
        # >> 중심(joint) 관절의 특징 채널에 각도 값을 할당한다.
        joint_angle_features[:, :, j_idx, 0] = angle
        

    # >> 3. 몸통 기준 상대 각도 (2D)
    # >> 3-1. 몸통 좌표계 기준 벡터 3개를 정의한다. (T, 2, 3)
    torso_Y_vec = coords[:, :, 20, :] - coords[:, :, 0, :]  # Y축 (위쪽)
    torso_X_vec = coords[:, :, 4, :] - coords[:, :, 8, :]  # X축 (오른쪽)
    torso_Z_vec = np.cross(torso_X_vec, torso_Y_vec)       # Z축 (정면)

    # >> 기준 벡터 정규화
    torso_Y_norm = torso_Y_vec / (np.linalg.norm(torso_Y_vec, axis=-1, keepdims=True) + 1e-8)
    torso_Z_norm = torso_Z_vec / (np.linalg.norm(torso_Z_vec, axis=-1, keepdims=True) + 1e-8)

    # >> 3-2. 4대 팔다리 벡터 정의 (T, 2, 3)
    vec_ru_arm = coords[:, :, 5, :] - coords[:, :, 4, :]  # 오른팔 (어깨->팔꿈치)
    vec_lu_arm = coords[:, :, 9, :] - coords[:, :, 8, :]  # 왼팔
    vec_r_thigh = coords[:, :, 13, :] - coords[:, :, 12, :] # 오른허벅지
    vec_l_thigh = coords[:, :, 17, :] - coords[:, :, 16, :] # 왼허벅지

    # >> 3-3. 각도 계산
    angle_ru_arm_Y = _calculate_angle_between_vectors(vec_ru_arm, torso_Y_norm) # (T, 2)
    angle_lu_arm_Y = _calculate_angle_between_vectors(vec_lu_arm, torso_Y_norm)
    angle_r_thigh_Y = _calculate_angle_between_vectors(vec_r_thigh, torso_Y_norm)
    angle_l_thigh_Y = _calculate_angle_between_vectors(vec_l_thigh, torso_Y_norm)
    
    angle_ru_arm_Z = _calculate_angle_between_vectors(vec_ru_arm, torso_Z_norm)
    angle_lu_arm_Z = _calculate_angle_between_vectors(vec_lu_arm, torso_Z_norm)
    angle_r_thigh_Z = _calculate_angle_between_vectors(vec_r_thigh, torso_Z_norm)
    angle_l_thigh_Z = _calculate_angle_between_vectors(vec_l_thigh, torso_Z_norm)

    # >> 3-4. 특징 채널에 할당 (2개의 새로운 채널 생성)
    rel_angle_Y_feat = np.zeros((T, 2, BASE_NUM_JOINTS, 1))
    rel_angle_Z_feat = np.zeros((T, 2, BASE_NUM_JOINTS, 1))
    
    rel_angle_Y_feat[:, :, 5, 0] = angle_ru_arm_Y   # 오른 팔꿈치
    rel_angle_Y_feat[:, :, 9, 0] = angle_lu_arm_Y   # 왼 팔꿈치
    rel_angle_Y_feat[:, :, 13, 0] = angle_r_thigh_Y # 오른 무릎
    rel_angle_Y_feat[:, :, 17, 0] = angle_l_thigh_Y # 왼 무릎
    
    rel_angle_Z_feat[:, :, 5, 0] = angle_ru_arm_Z
    rel_angle_Z_feat[:, :, 9, 0] = angle_lu_arm_Z
    rel_angle_Z_feat[:, :, 13, 0] = angle_r_thigh_Z
    rel_angle_Z_feat[:, :, 17, 0] = angle_l_thigh_Z

    # >> 4. 비-인접 관절 거리 (2D)
    # >> 4-1. 거리 계산
    dist_hands_vec = coords[:, :, 7, :] - coords[:, :, 11, :] # 오른손 - 왼손
    dist_feet_vec = coords[:, :, 15, :] - coords[:, :, 19, :] # 오른발 - 왼발
    dist_rh_rf_vec = coords[:, :, 7, :] - coords[:, :, 15, :] # 오른손 - 오른발
    dist_lh_lf_vec = coords[:, :, 11, :] - coords[:, :, 19, :] # 왼손 - 왼발

    # >> (T, 2)
    dist_hands = np.linalg.norm(dist_hands_vec, axis=-1)
    dist_feet = np.linalg.norm(dist_feet_vec, axis=-1)
    dist_rh_rf = np.linalg.norm(dist_rh_rf_vec, axis=-1)
    dist_lh_lf = np.linalg.norm(dist_lh_lf_vec, axis=-1)

    # >> 4-2. 척추 길이로 정규화
    # 0으로 나누는 것을 방지하는 '안전한' 척추 길이로 나눈다.
    norm_dist_hands = dist_hands / safe_torso_lengths_squeezed
    norm_dist_feet = dist_feet / safe_torso_lengths_squeezed
    norm_dist_rh_rf = dist_rh_rf / safe_torso_lengths_squeezed
    norm_dist_lh_lf = dist_lh_lf / safe_torso_lengths_squeezed

    # >> 4-3. 특징 채널에 할당 (2개의 새로운 채널 생성)
    inter_dist_feat_1 = np.zeros((T, 2, BASE_NUM_JOINTS, 1))
    inter_dist_feat_2 = np.zeros((T, 2, BASE_NUM_JOINTS, 1))

    # >> (T, 2) -> (T, 2, 1)로 브로드캐스팅 준비
    norm_dist_hands = norm_dist_hands[..., np.newaxis]
    norm_dist_feet = norm_dist_feet[..., np.newaxis]
    norm_dist_rh_rf = norm_dist_rh_rf[..., np.newaxis]
    norm_dist_lh_lf = norm_dist_lh_lf[..., np.newaxis]

    # >> 비대칭 할당 -> 대칭 할당
    inter_dist_feat_1[:, :, [7, 11], 0] = norm_dist_hands   # 오른손(7)과 왼손(11) 모두에 할당
    inter_dist_feat_1[:, :, [15, 19], 0] = norm_dist_feet  # 오른발(15)과 왼발(19) 모두에 할당
    inter_dist_feat_2[:, :, [7, 15], 0] = norm_dist_rh_rf  # 오른손(7)과 오른발(15) 모두에 할당
    inter_dist_feat_2[:, :, [11, 19], 0] = norm_dist_lh_lf # 왼손(11)과 왼발(19) 모두에 할당


    
    # >> 5. 두 명의 중심 거리 (1D)
    # >> 5-1. P0와 P1의 척추 하단(0번 관절) 좌표
    # coords shape은 (T, 2, 25, 3) 입니다
    p0_center = coords[:, 0, 0, :] # (T, 3)
    p1_center = coords[:, 1, 0, :] # (T, 3)

    # >> 5-2. 두 중심점 간의 거리 계산
    center_distance_vec = p0_center - p1_center
    center_distance = np.linalg.norm(center_distance_vec, axis=-1) # (T,)

    # >> 5-3. 각자의 '안전한' 척추 길이로 정규화
    # safe_torso_lengths_squeezed는 (T, 2) 입니다
    safe_torso_p0 = safe_torso_lengths_squeezed[:, 0] # (T,)
    safe_torso_p1 = safe_torso_lengths_squeezed[:, 1] # (T,)
    
    # 0으로 나누는 것을 방지 (safe_torso가 1e-8을 더했으므로 이미 안전함)
    # (T,) -> (T, 1)
    norm_center_dist_p0 = (center_distance / safe_torso_p0)[..., np.newaxis] 
    norm_center_dist_p1 = (center_distance / safe_torso_p1)[..., np.newaxis]

    # >> 5-4. 이 특징을 모든 관절(25개)에 브로드캐스팅
    # (T, 1) -> (T, 1, 1) -> (T, 25, 1)
    norm_center_dist_p0_broadcast = np.broadcast_to(norm_center_dist_p0[:, np.newaxis, :], (T, BASE_NUM_JOINTS, 1))
    norm_center_dist_p1_broadcast = np.broadcast_to(norm_center_dist_p1[:, np.newaxis, :], (T, BASE_NUM_JOINTS, 1))

    # >> 5-5. P0용 특징 배열과 P1용 특징 배열을 스택 (T, 2, 25, 1)
    interaction_feat_dist = np.stack([norm_center_dist_p0_broadcast, norm_center_dist_p1_broadcast], axis=1)


    # >> 6. P0-P1 손/발 상호작용 거리 (4D)
    # >> (T,) 형태의 거리 계산
    dist_rh_rh = np.linalg.norm(coords[:, 0, 7, :] - coords[:, 1, 7, :], axis=-1) # P0 오른손 - P1 오른손
    dist_lh_lh = np.linalg.norm(coords[:, 0, 11, :] - coords[:, 1, 11, :], axis=-1) # P0 왼손 - P1 왼손
    dist_rf_rf = np.linalg.norm(coords[:, 0, 15, :] - coords[:, 1, 15, :], axis=-1) # P0 오른발 - P1 오른발
    dist_lf_lf = np.linalg.norm(coords[:, 0, 19, :] - coords[:, 1, 19, :], axis=-1) # P0 왼발 - P1 왼발

    # >> (T,)
    safe_torso_p0 = safe_torso_lengths_squeezed[:, 0] 
    safe_torso_p1 = safe_torso_lengths_squeezed[:, 1]

    # >> (T, 2, 25, 1) 특징을 생성하는 내부 헬퍼 함수를 정의한다.
    def _create_inter_feat(dist_val, torso_p0, torso_p1):
        norm_dist_p0 = (dist_val / torso_p0)[..., np.newaxis]
        norm_dist_p1 = (dist_val / torso_p1)[..., np.newaxis]
        norm_dist_p0_b = np.broadcast_to(norm_dist_p0[:, np.newaxis, :], (T, BASE_NUM_JOINTS, 1))
        norm_dist_p1_b = np.broadcast_to(norm_dist_p1[:, np.newaxis, :], (T, BASE_NUM_JOINTS, 1))
        return np.stack([norm_dist_p0_b, norm_dist_p1_b], axis=1)

    # >> 4개의 새로운 특징 채널 생성
    inter_feat_rh_rh = _create_inter_feat(dist_rh_rh, safe_torso_p0, safe_torso_p1)
    inter_feat_lh_lh = _create_inter_feat(dist_lh_lh, safe_torso_p0, safe_torso_p1)
    inter_feat_rf_rf = _create_inter_feat(dist_rf_rf, safe_torso_p0, safe_torso_p1)
    inter_feat_lf_lf = _create_inter_feat(dist_lf_lf, safe_torso_p0, safe_torso_p1)
    
    # >> 7. 모든 특징 결합 (11D + 4D = 15D)
    combined_features_per_person = np.concatenate(
        (dynamic_features,      # 4D
         bone_length_features,  # 1D
         joint_angle_features,  # 1D
         rel_angle_Y_feat,      # 1D
         rel_angle_Z_feat,      # 1D
         inter_dist_feat_1,     # 1D
         inter_dist_feat_2,     # 1D
         interaction_feat_dist, # 1D
         inter_feat_rh_rh,      # 1D
         inter_feat_lh_lh,      # 1D
         inter_feat_rf_rf,      # 1D
         inter_feat_lf_lf       # 1D
        ), 
        axis=-1
    ) # shape: (T, 2, 25, 15)

    # >> 8. 최종 마스킹
    # 유효하지 않은(몸통 길이 < 1cm) 프레임/사람의 모든 특징을 0으로 설정한다.
    # valid_mask shape은 (T, 2, 1, 1)이며, 브로드캐스팅을 통해 (T, 2, 25, 15)에 적용된다.
    combined_features_per_person = combined_features_per_person * valid_mask

    
    person1_features = combined_features_per_person[:, 0, :, :]
    person2_features = combined_features_per_person[:, 1, :, :]
    
    # >> (T, 50, 15) 형태로 반환
    final_features = np.concatenate((person1_features, person2_features), axis=1)
    
    return final_features




## #------------------------------------------------------------------
# calculate_and_save_stats를 위한 일꾼 함수
## #------------------------------------------------------------------
def process_file_for_stats(filename):
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
    
    # >> 15차원 특징을 반환한다.
    features = _calculate_features(downsampled_coords)
    
    valid_frames = features.reshape(-1, features.shape[-1])
    valid_frames = valid_frames[np.abs(valid_frames).sum(axis=1) > 1e-6]
    
    if valid_frames.shape[0] > 0:
        # >> 특징 배열 전체를 반환하는 대신,
        # >> (개수, 합계, 제곱의 합)을 계산하여 반환합니다.
        count = valid_frames.shape[0]
        # >> (N, 15) -> (15,)
        sum_val = np.sum(valid_frames, axis=0) 
        # >> (N, 15) -> (15,)
        sum_sq_val = np.sum(np.square(valid_frames), axis=0) 
        
        return (count, sum_val, sum_sq_val)
    else:
        return None




# ## ------------------------------------------------------------------------------
# 훈련 데이터셋에 대해서만 특징을 계산하고,
# 전체 특징 데이터의 평균과 표준편차를 계산하여 파일로 저장한다.
# 이 통계치는 나중에 모델 훈련 시 데이터 정규화에 사용된다.
# ## ------------------------------------------------------------------------------
def calculate_and_save_stats():
    print("--- 1단계: 훈련 데이터셋 통계치 계산 시작 ---")
    # >>all_features 리스트 대신 누적 변수를 초기화한다.
    total_count = 0
    total_sum = np.zeros(config.NUM_COORDS)    # (15,) 크기 0배열
    total_sum_sq = np.zeros(config.NUM_COORDS) # (15,) 크기 0배열

    
    filenames = os.listdir(SOURCE_DATA_PATH)

    # >> 사용할 CPU 코어 수를 정한다.
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for stats calculation...")

    
    # >> 멀티프로세싱 Pool을 생성하고 작업을 분배한다.
    with Pool(processes=num_cores) as pool:
        # >> pool.imap_unordered를 사용하여 작업을 병렬 처리하고 tqdm으로 진행률을 표시한다.
        results_iterator = pool.imap_unordered(process_file_for_stats, filenames)

        # >> 반환된 결과(None이 아닌 것)를 all_features 리스트에 추가한다.
        for result in tqdm(results_iterator, total=len(filenames), desc="[Calculating Stats]"):
            if result is not None:
                # >> 튜플을 unpack한다.
                count, sum_val, sum_sq_val = result
                
                # >> 값을 누적한다.
                total_count += count
                total_sum += sum_val
                total_sum_sq += sum_sq_val

    if total_count == 0:
        print("치명적 오류: 유효한 훈련 데이터를 찾을 수 없습니다. 통계 계산 실패.")
        mean = np.zeros(config.NUM_COORDS)
        std_raw = np.ones(config.NUM_COORDS)
    else:
        # >> E[X] = 총합 / 개수
        mean = total_sum / total_count
        
        # >> E[X^2] = 제곱의 총합 / 개수
        mean_of_sq = total_sum_sq / total_count
        
        # >> Var(X) = E[X^2] - (E[X])^2 (분산)
        variance = mean_of_sq - np.square(mean)
        
        # >> StdDev(X) = sqrt(Var(X)) (표준편차)
        # >> 부동소수점 오류로 인해 variance가 아주 작은 음수가 되는 것을 방지한다.
        std_raw = np.sqrt(np.maximum(variance, 0.0))

    # >> 표준편차가 0에 가까운 값을 clip한다.
    epsilon = 1e-6
    std = np.clip(std_raw, a_min=epsilon, a_max=None)

    # >> 계산된 통계치를 .npz 파일로 저장한다.
    np.savez(STATS_FILE, mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1))
    print(f"통계치 계산 완료. '{STATS_FILE}' 파일에 저장되었습니다.")
    



## #------------------------------------------------------------------
# main 함수를 위한 일꾼 함수
## #------------------------------------------------------------------
def process_and_save_file(filename):
    """파일 하나를 전처리하고 .pt 파일로 저장합니다."""
    if not filename.endswith('.skeleton'):
        return

    skeleton_path = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(skeleton_path)
    
    if coords.shape[0] == 0:
        # >> 15차원 0벡터
        processed_features = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS)) 
        first_frame_coords = np.zeros((NUM_JOINTS, 3))
    else: 
        first_frame_raw = coords[0, :, :, :]
        first_frame_coords = np.concatenate((first_frame_raw[0], first_frame_raw[1]), axis=0)
        
        downsampled_coords = coords[::2, :, :]
        # >> 15차원 특징 계산
        raw_features = _calculate_features(downsampled_coords) 
        
        num_frames = raw_features.shape[0]
        if num_frames < MAX_FRAMES:
            pad_width = MAX_FRAMES - num_frames
            # >> 13차원 0벡터
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

    # >> 사용할 CPU 코어 수를 정한다.
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Using {num_cores} cores for multiprocessing...")

    # >> 멀티프로세싱 Pool을 사용하여 파일 저장을 병렬로 처리한다.
    with Pool(processes=num_cores) as pool:
        # >> list()로 감싸서 모든 작업이 끝날 때까지 기다린다.
        list(tqdm(pool.imap_unordered(process_and_save_file, filenames), total=len(filenames), desc="[Saving Raw Features]"))

    print("\n모든 데이터의 원본 특징(raw feature) 저장이 완료되었습니다.")


if __name__ == '__main__':
    main()
