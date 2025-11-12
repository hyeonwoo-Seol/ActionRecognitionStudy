# visualize_skeleton.py
#
# NTU-RGB+D .skeleton 파일을 읽어 3D로 시각화하는 스크립트입니다.
#
# 실행하려면 'matplotlib'이 필요합니다:
# pip install matplotlib

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm # _read_skeleton_file에서 사용 (pip install tqdm)

# ## --------------------------------------------------------------------------
# ## --- 시작: preprocess_ntu_data.py에서 복사한 코드 ---
# ##
# ## 원본 파일의 경로와 스켈레톤 정의, 파일 읽기 함수를 그대로 가져옵니다.
# ## --------------------------------------------------------------------------

# >> 처리할 NTU_RGB+D 60 skeleton 데이터가 있는 위치
# >> 중요: 이 경로가 현재 스크립트 위치 기준으로 올바른지 확인하세요!
# >> 원본 파일에서는 '../..' 였으므로, 이 스크립트의 위치에 따라 수정이 필요할 수 있습니다.
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
        print(f"Error: Could not read frame count from {filepath}")
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
                    print(f"Warning: Invalid body count at line {line_idx} in {filepath}. Skipping frame.")
                    continue

                for i in range(num_bodies):
                    line = f.readline() # body info line을 읽는다.
                    if not line: break
                    line_idx += 1
                    
                    body_info = line.strip().split() 
                    if len(body_info) < 1: 
                        print(f"Warning: Skipping empty body info line {line_idx} in {filepath}")
                        continue
                    
                    body_id = body_info[0] 

                    line = f.readline() # 관절 수가 적힌 다음줄을 읽는다.
                    if not line: break
                    line_idx += 1

                    try:
                        num_joints = int(line.strip())
                    except ValueError:
                        print(f"Warning: Invalid joint count at line {line_idx} in {filepath}.")
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
        print(f"Error during Pass 1 scan of {filepath}: {e}")
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
        print(f"Error during Pass 2 scan of {filepath}: {e}")
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    return final_coords

# ## --------------------------------------------------------------------------
# ## --- 끝: preprocess_ntu_data.py에서 복사한 코드 ---
# ## --------------------------------------------------------------------------


def visualize_single_frame(skeleton_file_path, frame_index=10, person_index=0):
    """
    .skeleton 파일을 읽어 특정 프레임의 3D 스켈레톤을 시각화합니다.
    """
    print(f"시각화 시도: {skeleton_file_path}")
    
    # 1. 파일 읽어오기
    coords = _read_skeleton_file(skeleton_file_path) # (T, 2, 25, 3)

    # 유효성 검사
    if coords.shape[0] == 0:
        print(f"오류: '{skeleton_file_path}' 파일에 유효한 데이터가 없습니다.")
        return

    if coords.shape[0] <= frame_index:
        print(f"경고: 요청한 프레임 {frame_index}가 없습니다. (총 {coords.shape[0]} 프레임)")
        print(f"대신 0번 프레임을 사용합니다.")
        frame_index = 0
        
    print(f"파일 로드 성공. 총 {coords.shape[0]} 프레임.")
    print(f"{frame_index}번 프레임, {person_index}번 사람 스켈레톤 시각화 중...")

    # 2. 특정 프레임 좌표 추출
    # (T, 2, 25, 3) -> (25, 3)
    joints_3d = coords[frame_index, person_index, :, :]
    
    # x, y, z 좌표 분리
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    z = joints_3d[:, 2]

    # 3. Matplotlib 3D 플롯 설정
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    

    # 4. (Step 4) 25개의 관절(joint)을 점으로 그리기
    ax.scatter(x, y, z, c='blue', marker='o', s=50, label='Joints')

    # (확인 사항 1) 관절 번호(0~24) 텍스트로 표시
    for i in range(joints_3d.shape[0]):
        # 관절 번호가 겹치지 않게 살짝 옆에 표시
        ax.text(x[i] + 0.01, y[i] + 0.01, z[i], f'{i}', color='black', fontsize=8)

    # 5. (Step 5) SKELETON_BONES 정보로 뼈(bone) 그리기
    # (확인 사항 2) 뼈 연결 정보 확인
    for parent, child in SKELETON_BONES:
        # 부모 관절 좌표
        p_coords = joints_3d[parent]
        # 자식 관절 좌표
        c_coords = joints_3d[child]
        
        # (0,0,0) 좌표로 연결되는 가짜 뼈는 그리지 않음
        if np.all(p_coords == 0) or np.all(c_coords == 0):
            continue
            
        # 두 점 (p_coords, c_coords)을 잇는 선 그리기
        ax.plot(
            [p_coords[0], c_coords[0]], # [x1, x2]
            [p_coords[1], c_coords[1]], # [y1, y2]
            [p_coords[2], c_coords[2]], # [z1, z2]
            'r-' # 빨간색 선
        )
        
    # (확인 사항 3) 좌표계 확인
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # 3D 뷰의 스케일을 동일하게 맞춰 왜곡을 방지
    # (가장 긴 축을 기준으로 다른 축의 범위를 동일하게 설정)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(f"NTU Skeleton 3D Viz\nFile: {os.path.basename(skeleton_file_path)}\nFrame: {frame_index}, Person: {person_index}")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    try:
        # SOURCE_DATA_PATH에서 .skeleton 파일 목록을 가져옵니다.
        filenames = os.listdir(SOURCE_DATA_PATH)
        skeleton_files = [f for f in filenames if f.endswith('.skeleton')]
        
        if not skeleton_files:
            print(f"오류: '{SOURCE_DATA_PATH}' 디렉터리에서 .skeleton 파일을 찾을 수 없습니다.")
            print("SOURCE_DATA_PATH 변수의 경로를 확인해주세요.")
        else:
            # 첫 번째 파일을 샘플로 선택합니다.
            # 다른 파일을 보려면 이 파일 이름을 변경하세요.
            # 예: sample_file_name = 'S001C001P001R001A050.skeleton' # 2인 행동
            sample_file_name = skeleton_files[0] 
            sample_file_path = os.path.join(SOURCE_DATA_PATH, sample_file_name)
            
            # 시각화 함수 호출
            # 10번 프레임, 0번 사람(주요 인물)
            visualize_single_frame(sample_file_path, frame_index=20, person_index=0)
            
            # 2인 행동(A050~A060) 파일의 경우 person_index=1로 변경하여 
            # 두 번째 사람도 확인해볼 수 있습니다.
            if 'A05' in sample_file_name or 'A06' in sample_file_name:
                 visualize_single_frame(sample_file_path, frame_index=20, person_index=1)

    except FileNotFoundError:
        print(f"오류: 데이터 경로를 찾을 수 없습니다: '{SOURCE_DATA_PATH}'")
        print("스크립트 상단의 SOURCE_DATA_PATH 변수를 올바른 NTU 데이터셋 경로로 수정해주세요.")
