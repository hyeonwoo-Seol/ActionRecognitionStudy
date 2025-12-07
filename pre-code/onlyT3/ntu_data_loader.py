# ntu_data_loader.py
# ## ----------------------------------------------------------------------
# class NTURGBDDataset 하나로 이루어진 파일로, 
# 데이터를 불러오고 훈련과 검증 모드를 설정하고 다양한 증강 기법을 적용시키는 파일이다.
# train.py에서 NTURGBDDataset 객체를 생성할 때 명시적으로 train, val을 지정한다.
# ## ----------------------------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import config

# X-Sub (피실험자)
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# X-View (카메라)
TRAINING_CAMERAS = [2, 3]
VALIDATION_CAMERAS = [1]

# ## ---------------------------------------------------------------------------------
# >> 특징(feature) 벡터가 15개의 채널로 구성됩니다.
# >> 기존 9D + 상대각도(2D) + 관절거리(2D)
# ## ---------------------------------------------------------------------------------
DIST_CH, DIR_CH = [0], list(range(1, 4))
BONE_LEN_CH, JOINT_ANG_CH = [4], [5]
REL_ANG_CH = [6, 7]
INTER_DIST_CH = [8, 9]
INTERACTION_CH = [10]
INTER_HAND_FOOT_CH = [11, 12, 13, 14]


# ## ----------------------------------------------------------------------------------
# PyTorch의 Dataset 클래스를 상속받아 NTU-RGB+D 데이터셋을 위한 커스텀 데이터 로더를 정의한다.
# 이 클래스는 데이터 전체를 메모리에 올리지 않고, __getitem__이 호출될 때마다 해당하는 파일을 하나씩
# 읽어와 처리하는 방식으로 동작하여 메모리를 효율적으로 사용한다..
# ## ----------------------------------------------------------------------------------
class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=300, protocol='xsub'):
        self.data_path = data_path # 전처리된 .pt 파일들이 저장되는 디렉토리 경로
        self.split = split         # 'train' 또는 'val' 모드를 설정
        self.max_frames = max_frames # 최대 프레임 수
        self.protocol = protocol
        self.all_subject_ids = list(range(1, 41))
        self.subject_to_index = {subject_id: index for index, subject_id in enumerate(self.all_subject_ids)}

        self.training_subjects = TRAINING_SUBJECTS


        self.samples = [] # 불러올 데이터 파일들의 전체 경로를 저장할 리스트
        self._load_data_path() # _load_data_path() 함수를 호출해서 self.samples 리스트를 채우기



        
        # >> 데이터 정규화에 사용할 평균과 표준편차 통계 파일('stats.npz')의 경로를 설정한다.
        stats_path = os.path.join(os.path.dirname(data_path.rstrip('/')), 'stats_allNew200.npz')
        if os.path.exists(stats_path):
            # >> 통계 파일이 존재하면 불러오기.
            stats = np.load(stats_path)


            # >> 불러온 통계치를 Pytorch 텐서로 변환해서 저장하기.
            self.mean = torch.from_numpy(stats['mean'].flatten()).float()
            self.std = torch.from_numpy(stats['std'].flatten()).float()
            print("Normalization stats loaded successfully.")
        else:
            print(f"FATAL: Statistics file not found at '{stats_path}'. Please run preprocess_ntu_data.py first.")
            # >> 오류 발생 시 평균 0, 표준편차 1을 사용한다.
            self.mean = torch.zeros(config.NUM_COORDS)
            self.std = torch.ones(config.NUM_COORDS)



            
    # >> 데이터 디렉토리를 스캔해서 'train', 'val' 모드에 맞는 파일 경로를 self.sampels 리스트에 저장한다.
    def _load_data_path(self):
        print(f"Scanning {self.split} data paths from '{self.data_path}'...")
        filenames = os.listdir(self.data_path)
        
        for filename in tqdm(filenames, desc=f"[{self.split.upper()}] Scanning file paths"):
            # >> .pt 확장자를 가진 파일만 처리한다.
            if not filename.endswith('.pt'):
                continue
                        
            # >> join 으로 운영체제에 맞게 자동으로 파일 경로를 합친다.
            file_path = os.path.join(self.data_path, filename)

            # 1. X-Sub 프로토콜
            if self.protocol == 'xsub':
                subject_id = int(filename[9:12]) # S001P001... -> P
                is_training_subject = subject_id in self.training_subjects

                if self.split == 'train' and is_training_subject:
                    self.samples.append(file_path)
                elif self.split == 'val' and not is_training_subject:
                    self.samples.append(file_path)

            # 2. X-View 프로토콜
            elif self.protocol == 'xview':
                camera_id = int(filename[5:8]) # S001C002... -> C
                if self.split == 'train' and camera_id in TRAINING_CAMERAS:
                    self.samples.append(file_path)
                elif self.split == 'val' and camera_id in VALIDATION_CAMERAS:
                    self.samples.append(file_path)
                            

    # >> 데이터셋의 총 샘플 수를 반환한다.
    def __len__(self):
        return len(self.samples)


    
    # >> 주어진 인덱스에 해당하는 데이터 샘플을 불러와 처리하고 반환한다.
    def __getitem__(self, index):
        file_path = self.samples[index]
        saved_data = torch.load(file_path)
        
        # >> 1. 원본(raw) 특징 로드. shape: (T, J, 9)
        features = saved_data['data'] 
        action_label = saved_data['label']

        filename = os.path.basename(file_path)

        if self.protocol == 'xsub':
            subject_id = int(filename[9:12])
            subject_id = int(filename[9:12]) # S001P001... -> P (1~40)
            subject_label = self.subject_to_index.get(subject_id, 0) # 0~39
        
        elif self.protocol == 'xview':
            camera_id = int(filename[5:8])
            if camera_id in TRAINING_CAMERAS:
                subject_label = 0 # '그룹 A' (훈련용 도메인)
            else:
                subject_label = 1 # '그룹 B' (검증용 도메인)
        else:
            subject_label = 0 # 기본값 (오류 방지)

                

        # >> 2. (훈련 시) 데이터 증강
        if self.split == 'train':
            
            # >> 2-1. 임의 회전
            if np.random.rand() > config.PROB:
                angle = np.random.uniform(-15, 15) * np.pi / 180.0
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                
                rotation_matrix = torch.tensor([
                    [cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]
                ], dtype=torch.float32)

                
                # >> 13D 특징을 스칼라와 3D 벡터로 분리
                # >> 스칼라 특징 (회전 안 함)
                f_dist = features[..., DIST_CH]     # (T, J, 1)
                f_len = features[..., BONE_LEN_CH]  # (T, J, 1)
                f_ang = features[..., JOINT_ANG_CH] # (T, J, 1)
                f_rel_ang = features[..., REL_ANG_CH] # (T, J, 2)
                f_inter_dist = features[..., INTER_DIST_CH] # (T, J, 2)
                f_interaction = features[..., INTERACTION_CH] # (T, J, 1)
                f_inter_hand_foot = features[..., INTER_HAND_FOOT_CH] # (T, J, 4)
                
                # >> 3D 벡터 특징 (회전 함)
                f_dir = features[..., DIR_CH]       # (T, J, 3)
                # f_acc = features[..., ACC_CH]       # (T, J, 3)

                # (T, J, 3) -> (T*J, 3)
                f_dir_flat = f_dir.reshape(-1, 3) 
                # 회전 적용
                f_dir_rotated_flat = f_dir_flat @ rotation_matrix.T
                # (T*J, 3) -> (T, J, 3) 복원
                f_dir_rotated = f_dir_rotated_flat.reshape_as(f_dir)

                
                # f_acc_rotated = rotatable_rotated[..., 3:6] # (T, J, 3)
                
                # >> 모든 특징을 원래 순서대로 다시 결합 (11D)
                features = torch.cat([
                    f_dist, f_dir_rotated, f_len, f_ang, # f_acc_rotated 제거
                    f_rel_ang, f_inter_dist, f_interaction,
                    f_inter_hand_foot
                ], dim=-1) # (T, J, 15)

              
            # >> 2-2. 가우시안 노이즈 추가 (원본과 동일)
            if np.random.rand() > config.PROB:
                noise = torch.randn_like(features) * 0.005
                features += noise

                
            # >> 2-3. 임의 스케일링
            if np.random.rand() > config.PROB:
                scale_factor = np.random.uniform(0.8, 1.2)

                # >> 모든 거리(Distance) 기반 특징 채널을 정의합니다.
                scale_channels = DIST_CH + BONE_LEN_CH + INTER_DIST_CH + INTERACTION_CH + INTER_HAND_FOOT_CH
                
                # >> 거리(채널 0)에 스케일링
                features[..., scale_channels] *= scale_factor
                
                
            # >> 2-4. 관절 마스킹 (원본과 동일)
            if np.random.rand() > config.PROB:
                num_joints = features.shape[1]
                num_joints_to_mask = np.random.randint(2, 6) 
                joint_indices_to_mask = np.random.choice(num_joints, num_joints_to_mask, replace=False)
                features[:, joint_indices_to_mask, :] = 0.0

                
            # >> 2-5. 시간 마스킹 (원본과 동일)
            if np.random.rand() > config.PROB:
                total_frames = features.shape[0]
                max_mask_len = 20
                if total_frames > max_mask_len:
                    mask_len = np.random.randint(5, max_mask_len + 1)
                    start_frame = np.random.randint(0, total_frames - mask_len)
                    features[start_frame : start_frame + mask_len, :, :] = 0.0


        # >> 3. 정규화 적용 (self.mean/std가 13D이므로 자동으로 13D로 적용됨)
        std_eps = self.std + 1e-8
        features = (features - self.mean) / std_eps


        # >> 4. SlowFast 구현하기
        # >> 4.1 Fast Path
        data_fast = features.permute(2, 0, 1)

        # >> 4.2 Slow Fast
        data_slow = data_fast[:, ::2, :]

        # >> 4.3 Return
        return data_fast, data_slow, action_label, subject_label
