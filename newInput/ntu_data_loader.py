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
    def __init__(self, data_path, split='train', max_frames=config.MAX_FRAMES, protocol='xsub'):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        self.protocol = protocol
        self.training_subjects = TRAINING_SUBJECTS
        
        # 샘플 로드
        self.samples = []
        self._load_data_path()

        # 통계치 로드
        base_dir = os.path.dirname(data_path.rstrip('/'))
        
        if self.protocol == 'xsub':
            stats_filename = 'stats_xsub_SKF.npz'
        elif self.protocol == 'xview':
            stats_filename = 'stats_xview_SKF.npz'
        else:
            stats_filename = 'stats_xsub_SKF.npz'
            
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
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist.")
            return

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
        features = data['data'] # (MAX_FRAMES, 50, 12)
        action_label = data['label']
        filename = os.path.basename(self.samples[index])
        
        
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
                if T > mask_len:
                    start = np.random.randint(0, T - mask_len)
                    features[start:start+mask_len] = 0

        # Normalization
        features = (features - self.mean) / (self.std + 1e-8)
        
        # SlowFast Split
        data_fast = features.permute(2, 0, 1) # (C, T, J)
        data_slow = data_fast.clone() 
        
        return data_fast, data_slow, action_label, aux_label
