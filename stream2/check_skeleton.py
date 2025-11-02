import os
import random
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import config  # config.py 파일에서 설정을 불러옵니다.

# --- 설정 ---

# 1. config.py에서 .pt 파일이 저장된 경로를 가져옵니다.
DATA_DIR = config.DATASET_PATH 

# 2. 확인할 상호작용 클래스 ID (A050 ~ A060)
INTERACTION_CLASS_IDS = [f"A{i:03d}" for i in range(50, 61)]

# --- 시각화 함수 ---

def visualize_random_interaction(data_dir, class_ids):
    """
    지정된 디렉토리에서 상호작용 클래스 .pt 파일을 무작위로 하나 선택하여
    첫 프레임의 3D 스켈레톤을 시각화합니다.
    """
    
    print(f"Scanning for files in: {data_dir}")
    
    # 1. 디렉토리 내 모든 .pt 파일을 스캔합니다.
    try:
        all_files = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found at '{data_dir}'")
        print("config.py의 DATASET_PATH가 올바른지 확인해주세요.")
        return

    # 2. 상호작용 클래스 ID(A050~A060)를 포함하는 파일만 필터링합니다.
    interaction_files = [
        f for f in all_files 
        if any(class_id in f for class_id in class_ids) and f.endswith('.pt')
    ]
    
    if not interaction_files:
        print(f"Error: No interaction class files (A050-A060) found in {data_dir}")
        return

    # 3. 필터링된 파일 중 하나를 무작위로 선택합니다.
    target_file = "S007C002P008R001A050.pt"
    file_path = os.path.join(data_dir, target_file)
    print(f"--- Loading sample file: {target_file} ---")

    # 4. .pt 파일을 로드하고 'first_frame_coords' 텐서를 추출합니다.
    try:
        data = torch.load(file_path, map_location='cpu')
        # (50, 3) 형태의 텐서를 numpy 배열로 변환
        coords = data['first_frame_coords'].cpu().numpy()
    except Exception as e:
        print(f"Error loading or processing file {file_path}: {e}")
        return

    if coords.shape != (50, 3):
        print(f"Error: Expected 'first_frame_coords' shape (50, 3), but got {coords.shape}")
        return

    # 5. Person 1과 Person 2의 좌표를 분리합니다.
    p1_coords = coords[0:25, :]  # (25, 3)
    p2_coords = coords[25:50, :] # (25, 3)

    # 6. (중요) Person 2 데이터가 유효한지 확인합니다.
    # np.any(p2_coords != 0)는 p2_coords에 0이 아닌 값이 하나라도 있는지 확인
    p2_is_valid = np.any(p2_coords != 0)
    
    print(f"Person 2 Data Valid (non-zero found): {p2_is_valid}")
    if not p2_is_valid:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Warning: Person 2 데이터가 모두 0입니다.")
        print("이 샘플은 두 번째 사람이 로드되지 않았거나 엑스트라가 잘못 로드되었을 수 있습니다.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # 7. Matplotlib을 사용하여 3D 스켈레톤을 시각화합니다.
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Person 1 (파란색)
    ax.scatter(p1_coords[:, 0], p1_coords[:, 1], p1_coords[:, 2], c='blue', label='Person 1', s=50)
    
    # Person 2 (빨간색)
    ax.scatter(p2_coords[:, 0], p2_coords[:, 1], p2_coords[:, 2], c='red', label='Person 2 (Valid: '+str(p2_is_valid)+')', s=50)

    # (선택 사항) 관절 번호 표시. 너무 복잡하면 이 4줄을 주석 처리(#)하세요.
    # for i in range(25):
    #     ax.text(p1_coords[i, 0], p1_coords[i, 1], p1_coords[i, 2], f'p1_{i}', color='blue', fontsize=8)
    # for i in range(25):
    #     ax.text(p2_coords[i, 0], p2_coords[i, 1], p2_coords[i, 2], f'p2_{i}', color='red', fontsize=8)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"First Frame Skeleton Visualization\n{target_file}")
    
    # 축 스케일을 동일하게 맞춰 실제 비율을 볼 수 있게 함
    max_range = np.array([p1_coords.max(0)-p1_coords.min(0), p2_coords.max(0)-p2_coords.min(0)]).max() / 2.0
    mid_x = np.mean(coords[:,0])
    mid_y = np.mean(coords[:,1])
    mid_z = np.mean(coords[:,2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()

    def on_key_press(event):
        """아무 키나 누르면 plt.close()를 호출하여 창을 닫습니다."""
        plt.close()

    # 2. figure의 캔버스에 'key_press_event'와 on_key_press 함수를 연결합니다.
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    plt.show()


# --- 스크립트 실행 ---

if __name__ == "__main__":
    visualize_random_interaction(DATA_DIR, INTERACTION_CLASS_IDS)
