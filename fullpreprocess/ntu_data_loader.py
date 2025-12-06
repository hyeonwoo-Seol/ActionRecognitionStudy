# ntu_data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import config

# X-Sub
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# X-View
TRAINING_CAMERAS = [2, 3]
VALIDATION_CAMERAS = [1]

class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=300, protocol='xsub'):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        self.protocol = protocol
        self.training_subjects = TRAINING_SUBJECTS
        
        # 샘플 로드
        self.samples = []
        self._load_data_path()

        # [수정] 통계치 로드 (Protocol에 따라 분기)
        base_dir = os.path.dirname(data_path.rstrip('/'))
        
        if self.protocol == 'xsub':
            stats_filename = 'stats_xsub.npz'
        elif self.protocol == 'xview':
            stats_filename = 'stats_xview.npz'
        else:
            stats_filename = 'stats_12D_Norm.npz'
            
        stats_path = os.path.join(base_dir, stats_filename)
        
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean = torch.from_numpy(stats['mean'].flatten()).float()
            self.std = torch.from_numpy(stats['std'].flatten()).float()
            print(f"[{split.upper()}] Normalization stats loaded from {stats_filename} (Protocol: {protocol}).")
        else:
            print(f"Warning: Stats not found at {stats_path}. Using identity.")
            self.mean = torch.zeros(config.NUM_COORDS)
            self.std = torch.ones(config.NUM_COORDS)

    def _load_data_path(self):
        filenames = sorted(os.listdir(self.data_path))
        for filename in filenames:
            if not filename.endswith('.pt'): continue
            
            # Protocol check
            if self.protocol == 'xsub':
                sid = int(filename[9:12])
                is_train = sid in self.training_subjects
            elif self.protocol == 'xview':
                cid = int(filename[5:8])
                is_train = cid in TRAINING_CAMERAS
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
            
            if (self.split == 'train' and is_train) or (self.split == 'val' and not is_train):
                self.samples.append(os.path.join(self.data_path, filename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = torch.load(self.samples[index])
        features = data['data'] # (T_raw, 50, 12)
        action_label = data['label']
        
        filename = os.path.basename(self.samples[index])
        
        # [수정 1] 데이터 길이 맞추기 (Truncate or Pad)
        # 로드된 데이터가 설정된 MAX_FRAMES보다 길면 자르고, 짧으면 0으로 채웁니다.
        T_raw = features.shape[0]
        if T_raw > config.MAX_FRAMES:
            features = features[:config.MAX_FRAMES, :, :]
        elif T_raw < config.MAX_FRAMES:
            pad_len = config.MAX_FRAMES - T_raw
            pad = torch.zeros((pad_len, features.shape[1], features.shape[2]), dtype=features.dtype)
            features = torch.cat((features, pad), dim=0)
            
        # 이제 features의 길이는 무조건 config.MAX_FRAMES (100)이 됩니다.

        # GRL 타겟 라벨 설정
        aux_label = 0
        if self.protocol == 'xsub':
            sid = int(filename[9:12])
            aux_label = sid - 1 
        elif self.protocol == 'xview':
            cid = int(filename[5:8])
            aux_label = cid - 1 
        
        # --- Data Augmentation ---
        if self.split == 'train':
            # 1. Rotation 
            if np.random.rand() < config.PROB:
                angle = np.random.uniform(-15, 15) * np.pi / 180.0
                c, s = np.cos(angle), np.sin(angle)
                rot_mat = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
                
                # [수정 2] config.MAX_FRAMES 대신 현재 텐서의 shape[0]을 사용하여 안전하게 변환
                curr_T = features.shape[0]
                reshaped = features.view(curr_T, config.NUM_JOINTS, 4, 3)
                rotated = torch.matmul(reshaped, rot_mat.T)
                features = rotated.view(curr_T, config.NUM_JOINTS, 12)

            # 2. Scaling
            if np.random.rand() < config.PROB:
                scale = np.random.uniform(0.9, 1.1)
                features *= scale
                
            # 3. Time Masking
            if np.random.rand() < config.PROB:
                T = features.shape[0]
                mask_len = np.random.randint(5, 20)
                start = np.random.randint(0, T - mask_len)
                features[start:start+mask_len] = 0

        # Normalization
        features = (features - self.mean) / (self.std + 1e-8)
        
        # SlowFast Split
        data_fast = features.permute(2, 0, 1) # (C, T, J)
        
        # [수정] 모델 내부에서 다운샘플링을 2번 수행하므로 로더에서는 원본 길이 유지
        data_slow = data_fast.clone() 
        
        return data_fast, data_slow, action_label, aux_label
