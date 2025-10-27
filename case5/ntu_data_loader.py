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


# >> 특징(feature) 벡터는 7개의 채널로 구성됩니다: 거리(1), 방향(3), 가속도(3).
# >> 데이터 증강 시 특정 채널에만 선택적으로 연산을 적용하기 위해 각 채널의 인덱스를 상수로 정의한다.
DIST_CH, DIR_CH, ACC_CH = [0], list(range(1, 4)), list(range(4, 7))



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
        stats_path = os.path.join(os.path.dirname(data_path.rstrip('/')), 'stats_4fps.npz')
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
        # >> 인덱스에 해당하는 파일 경로를 가져온다.
        file_path = self.samples[index]

        
        # >> torch.load를 이용해 전처리된 .pt 파일을 불러온다.
        saved_data = torch.load(file_path)

        
        # >> 1. 정규화되지 않은 원본(raw) 특징 로드
        features = saved_data['data'] # shape: (T, J, 7)
        label = saved_data['label']
        

        
        # >> 2. (훈련 시) 데이터 증강을 원본 특징에 바로 적용
        if self.split == 'train':
            # >> 2-1. 임의 회전
            if np.random.rand() > config.PROB:
                # >> -10도에서 +10도 사이의 각도를 무작위로 선택한다.
                angle = np.random.uniform(-10, 10) * np.pi / 180.0
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)

                
                # >> Y축 회전 행렬을 생성한다.
                rotation_matrix = torch.tensor([
                    [cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]
                ], dtype=torch.float32)

                
                # >> 채널 0 (거리)는 회전 안 한다.
                non_rotatable = features[..., 0:1]

                
                # >> 채널 1부터 끝 (3D방향, 3D가속도)은 회전한다.
                rotatable = features[..., 1:]

                
                # >> (..., 6) 형태의 벡터를 (..., 2, 3)으로 보고 각 3D 벡터에 회전 행렬을 적용한다.
                rot_flat = rotatable.reshape(-1, 3) # (T*J*2, 3) 형태로 변환
                rot_rotated = rot_flat @ rotation_matrix.T # 회전 적용 
                rotatable_rotated = rot_rotated.reshape_as(rotatable) # 원래 형태로 복원 

                
                # >> 회전된 벡터와 회전하지 않은 스칼라 값을 다시 합친다.
                features = torch.cat([non_rotatable, rotatable_rotated], dim=-1)

              
            # >> 2-2. 가우시안 노이즈 추가
            if np.random.rand() > config.PROB:
                noise = torch.randn_like(features) * 0.005 # 노이즈 수준
                features += noise

                
            # 2-3. 임의 스케일링
            if np.random.rand() > config.PROB:
                scale_factor = np.random.uniform(0.9, 1.1)

                
                # >> 거리(0번 채널)와 가속도(3,4,5번 채널)에만 스케일링 적용
                # >> 거리(채널 0)에 스케일링
                features[..., DIST_CH] *= scale_factor


                # >> 3D 방향(채널 1,2,3)은 단위 벡터이므로 제외
                # >> 3D 가속도(채널 4,5,6)에 스케일링
                features[..., ACC_CH] *= scale_factor

                
            # >> 2-4. 관절 마스킹
            # >> 모델이 부분적인 정보만으로도 동작하도록 훈련한다.
            if np.random.rand() > config.PROB:
                num_joints = features.shape[1]

                
                # >> 1개에서 3개의 관절을 무작위로 선택해서 마스킹한다.
                num_joints_to_mask = np.random.randint(1, 4) 
                joint_indices_to_mask = np.random.choice(num_joints, num_joints_to_mask, replace=False)

                
                # >> 선택된 관절의 모든 시간(T)과 특징(C) 값을 0으로 설정한다.
                features[:, joint_indices_to_mask, :] = 0.0

                
            # >> 2-5. 시간 마스킹
            # >> 모델이 시간적 연속성이 일부 깨져도 동작하도록 훈련한다.
            if np.random.rand() > config.PROB:
                total_frames = features.shape[0] # 전체 프레임 수 (T)
                max_mask_len = 20 # 마스킹할 최대 프레임 길이

                
                # >>프레임 수가 최대 마스킹 길이보다 클 때만 적용한다.
                if total_frames > max_mask_len:
                    # >> 5에서 20 프레임 사이의 길이를 무작위로 선택한다.
                    mask_len = np.random.randint(5, max_mask_len + 1)

                    
                    # >> 마스킹을 시작할 프레임 위치를 무작위로 선택한다.
                    start_frame = np.random.randint(0, total_frames - mask_len)

                    
                    # >> 선택된 시간 구간의 모든 관절(J)과 특징(C) 값을 0으로 설정한다.
                    features[start_frame : start_frame + mask_len, :, :] = 0.0


                    
        # >> 3. 증강이 끝난 데이터에 정규화 적용
        # >> (X - 평균) / 표준편차를 해서 훈련 데이터의 분포를 안정시켜 학습한다.
        std_eps = self.std + 1e-8
        features = (features - self.mean) / std_eps


        
        # >> 4. 최종 텐서 형태로 변환
        data = features.permute(2, 0, 1) # (T, J, C) -> (C, T, J)

        return data, label
