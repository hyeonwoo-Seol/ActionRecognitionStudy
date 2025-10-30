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


# ## ---------------------------------------------------------------------------------
# >> [수정됨] 특징(feature) 벡터가 9개의 채널로 구성됩니다.
# >> 거리(1), 방향(3), 가속도(3), 뼈길이(1), 관절각도(1)
# ## ---------------------------------------------------------------------------------
DIST_CH, DIR_CH, ACC_CH = [0], list(range(1, 4)), list(range(4, 7))
BONE_LEN_CH, JOINT_ANG_CH = [7], [8]



# ## ----------------------------------------------------------------------------------
# PyTorch의 Dataset 클래스를 상속받아 NTU-RGB+D 데이터셋을 위한 커스텀 데이터 로더를 정의한다.
# 이 클래스는 데이터 전체를 메모리에 올리지 않고, __getitem__이 호출될 때마다 해당하는 파일을 하나씩
# 읽어와 처리하는 방식으로 동작하여 메모리를 효율적으로 사용한다..
# ## ----------------------------------------------------------------------------------
class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=300):
        self.data_path = data_path # 전처리된 .pt 파일들이 저장되는 디렉토리 경로
        self.split = split         # 'train' 또는 'val' 모드를 설정
        self.max_frames = max_frames # 최대 프레임 수


        # >> 훈련 데이터셋을 구성하는 subject의 ID 목록
        # >> 이 목록을 사용해서 train set 과 val set을 분리한다.
        self.training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]


        self.samples = [] # 불러올 데이터 파일들의 전체 경로를 저장할 리스트
        self._load_data_path() # _load_data_path() 함수를 호출해서 self.samples 리스트를 채우기



        
        # >> 데이터 정규화에 사용할 평균과 표준편차 통계 파일('stats.npz')의 경로를 설정한다.
        stats_path = os.path.join(os.path.dirname(data_path.rstrip('/')), 'stats.npz')
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

            
            # >> 파일명에서 피실험자 ID를 정수형으로 추출한다.
            subject_id = int(filename[9:12])

            
            # >> 추출한 ID가 훈련용 ID 목록에 포함되어 있는지 확인한다.
            is_training_subject = subject_id in self.training_subjects

            
            # >> join 으로 운영체제에 맞게 자동으로 파일 경로를 합친다.
            file_path = os.path.join(self.data_path, filename)

            
            # >> 현재 모드에 따라 파일 경로를 self.samples 리스트에 추가한다.
            if self.split == 'train' and is_training_subject: # train 모드 
                self.samples.append(file_path)
            elif self.split == 'val' and not is_training_subject: # val 모드
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
        label = saved_data['label']
        

        # >> 2. (훈련 시) 데이터 증강
        if self.split == 'train':
            
            # >> 2-1. 임의 회전 [로직 수정됨]
            if np.random.rand() > config.PROB:
                angle = np.random.uniform(-10, 10) * np.pi / 180.0
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                
                rotation_matrix = torch.tensor([
                    [cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]
                ], dtype=torch.float32)

                
                # >> [수정됨] 9D 특징을 스칼라와 3D 벡터로 분리
                # >> 스칼라 특징 (회전 안 함)
                f_dist = features[..., DIST_CH]     # (T, J, 1)
                f_len = features[..., BONE_LEN_CH]  # (T, J, 1)
                f_ang = features[..., JOINT_ANG_CH] # (T, J, 1)

                # >> 3D 벡터 특징 (회전 함)
                f_dir = features[..., DIR_CH]       # (T, J, 3)
                f_acc = features[..., ACC_CH]       # (T, J, 3)
                
                # >> 회전할 벡터들을 하나로 합침 (T, J, 6)
                rotatable = torch.cat([f_dir, f_acc], dim=-1)

                
                # >> (..., 6) 형태의 벡터를 (..., 2, 3)으로 보고 각 3D 벡터에 회전 행렬을 적용
                rot_flat = rotatable.reshape(-1, 3) # (T*J*2, 3) 형태로 변환
                rot_rotated = rot_flat @ rotation_matrix.T # 회전 적용 
                rotatable_rotated = rot_rotated.reshape_as(rotatable) # (T, J, 6) 복원

                # >> 회전된 벡터를 다시 분리
                f_dir_rotated = rotatable_rotated[..., 0:3] # (T, J, 3)
                f_acc_rotated = rotatable_rotated[..., 3:6] # (T, J, 3)
                
                # >> [수정됨] 모든 특징을 원래 순서대로 다시 결합
                features = torch.cat([
                    f_dist, f_dir_rotated, f_acc_rotated, f_len, f_ang
                ], dim=-1) # (T, J, 9)

              
            # >> 2-2. 가우시안 노이즈 추가 (원본과 동일)
            if np.random.rand() > config.PROB:
                noise = torch.randn_like(features) * 0.005
                features += noise

                
            # >> 2-3. 임의 스케일링 [로직 수정됨]
            if np.random.rand() > config.PROB:
                scale_factor = np.random.uniform(0.9, 1.1)
                
                # >> 거리(채널 0)에 스케일링
                features[..., DIST_CH] *= scale_factor
                
                # >> 3D 방향(채널 1,2,3)은 단위 벡터이므로 제외
                
                # >> 3D 가속도(채널 4,5,6)에 스케일링
                features[..., ACC_CH] *= scale_factor

                # >> [수정됨] 뼈 길이(채널 7)에도 스케일링 적용
                features[..., BONE_LEN_CH] *= scale_factor

                # >> 관절 각도(채널 8)는 스케일링 제외
                
                
            # >> 2-4. 관절 마스킹 (원본과 동일)
            if np.random.rand() > config.PROB:
                num_joints = features.shape[1]
                num_joints_to_mask = np.random.randint(1, 4) 
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


        # >> 3. 정규화 적용 (원본과 동일, self.mean/std가 9D이므로 자동으로 9D로 적용됨)
        std_eps = self.std + 1e-8
        features = (features - self.mean) / std_eps


        
        # >> 4. 최종 텐서 형태로 변환 (원본과 동일)
        data = features.permute(2, 0, 1) # (T, J, C) -> (C, T, J)

        return data, label