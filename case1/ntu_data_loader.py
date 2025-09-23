# ntu_data_loader.py (효율적으로 수정됨)

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import config

DIST_CH, DIR_CH, ACC_CH = [0], list(range(1, 4)), list(range(4, 7))


class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=300):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        self.training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        self.samples = []
        self._load_data()

        
        stats_path = os.path.join(os.path.dirname(data_path.rstrip('/')), 'stats.npz')
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean = torch.from_numpy(stats['mean'].flatten()).float()
            self.std = torch.from_numpy(stats['std'].flatten()).float()
            print("Normalization stats loaded successfully.")
        else:
            print(f"FATAL: Statistics file not found at '{stats_path}'. Please run preprocess_ntu_data.py first.")
            self.mean = torch.zeros(config.NUM_COORDS)
            self.std = torch.ones(config.NUM_COORDS)


    def _load_data(self):
        print(f"Loading {self.split} data into memory from '{self.data_path}'...")
        filenames = os.listdir(self.data_path)
        
        for filename in tqdm(filenames, desc=f"[{self.split.upper()}] Loading data into RAM"):
            if not filename.endswith('.pt'):
                continue

            subject_id = int(filename[9:12])
            is_training_subject = subject_id in self.training_subjects
            
            file_path = os.path.join(self.data_path, filename)

            # 파일 경로가 아닌, 실제 데이터를 self.samples 리스트에 저장
            if self.split == 'train' and is_training_subject:
                # torch.load를 여기서 호출하여 데이터를 미리 읽음
                data = torch.load(file_path)
                self.samples.append(data)
            elif self.split == 'val' and not is_training_subject:
                # torch.load를 여기서 호출하여 데이터를 미리 읽음
                data = torch.load(file_path)
                self.samples.append(data)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        saved_data = self.samples[index]
        
        # 1. 정규화되지 않은 원본(raw) 특징 로드
        features = saved_data['data'] # shape: (T, J, 7)
        label = saved_data['label']
        first_frame_coords = saved_data['first_frame_coords'] # shape: (J, 3)
        
        # 2. (훈련 시) 데이터 증강을 원본 특징에 바로 적용
        if self.split == 'train':
            # 2-1. 임의 회전 (50% 확률)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10) * np.pi / 180.0
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)
            
                rotation_matrix = torch.tensor([
                    [cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]
                ], dtype=torch.float32)
                
                # 채널 0 (거리)는 회전 안 함
                non_rotatable = features[..., 0:1]
                # 채널 1부터 끝 (3D방향, 3D가속도)은 회전
                rotatable = features[..., 1:]
                
                # 회전 로직은 3D 벡터 2개가 합쳐진 (..., 6) 형태에서도 잘 동작함
                rot_flat = rotatable.reshape(-1, 3)
                rot_rotated = rot_flat @ rotation_matrix.T
                rotatable_rotated = rot_rotated.reshape_as(rotatable)
            
            
                features = torch.cat([non_rotatable, rotatable_rotated], dim=-1)

            # 2-2. 가우시안 노이즈 추가 (50% 확률)
            if np.random.rand() > 0.5:
                noise = torch.randn_like(features) * 0.005 # 노이즈 수준(level)은 조절 가능
                features += noise

            # 2-3. 임의 스케일링 (50% 확률)
            if np.random.rand() > 0.5:
                scale_factor = np.random.uniform(0.9, 1.1)
                
                # 거리(0번 채널)와 가속도(3,4,5번 채널)에만 스케일링 적용
                # 거리(채널 0)에 스케일링
                features[..., DIST_CH] *= scale_factor
                # 3D 방향(채널 1,2,3)은 단위 벡터이므로 제외
                # 3D 가속도(채널 4,5,6)에 스케일링
                features[..., ACC_CH] *= scale_factor

            # 2-4. 관절 마스킹
            if np.random.rand() > 0.5:
                num_joints = features.shape[1]

                # 1개에서 3개의 관절을 무작위로 선택해서 마스킹
                num_joints_to_mask = np.random.randint(1, 4) 
                joint_indices_to_mask = np.random.choice(num_joints, num_joints_to_mask, replace=False)

                # 선택된 관절의 모든 시간(T)과 특징(C) 값을 0으로 설정
                features[:, joint_indices_to_mask, :] = 0.0

            # 2-5. 시간 마스킹
            if np.random.rand() > 0.5:
                total_frames = features.shape[0] # 전체 프레임 수 (T)
                max_mask_len = 20 # 마스킹할 최대 프레임 길이

                # 프레임 수가 최대 마스킹 길이보다 클 때만 적용
                if total_frames > max_mask_len:
                    # 5에서 20 프레임 사이의 길이를 무작위로 선택
                    mask_len = np.random.randint(5, max_mask_len + 1)
                    # 마스킹을 시작할 프레임 위치를 무작위로 선택
                    start_frame = np.random.randint(0, total_frames - mask_len)
                    
                    # 선택된 시간 구간의 모든 관절(J)과 특징(C) 값을 0으로 설정
                    features[start_frame : start_frame + mask_len, :, :] = 0.0

        # 3. 증강이 끝난 데이터에 정규화 적용
        std_eps = self.std + 1e-8
        features = (features - self.mean) / std_eps
            
        # 4. 최종 텐서 형태로 변환
        data = features.permute(2, 0, 1) # (C, T, J)

        return data, label, first_frame_coords
