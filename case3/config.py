# config.py

import torch

# NTU RGB+D 60 데이터셋의 .np 파일들이 있는 디렉토리 경로
# 이 경로에는 preprocess_ntu_data.py를 실행하고 생성된 .np 파일들이 있는 경로로 지정해야 한다.
DATASET_PATH = 'nturgbd_processed/'

# 학습된 모델 가중치(체크포인트)를 저장할 디렉토리
SAVE_DIR = 'checkpoints/'

# 재현성을 위해 시드 설정
SEED = 42


# --- 데이터 로더 설정 ---
MAX_FRAMES = 150  # 시퀀스의 최대 길이 (패딩 또는 절단 기준)
BATCH_SIZE = 32   # 배치 크기
NUM_WORKERS = 6
PIN_MEMORY = True

# --- 모델 하이퍼파라미터 ---
NUM_JOINTS = 50   # 관절 수
NUM_COORDS = 7    # 거리1 + 방향3 + 가속도3
D_MODEL = 256     
NUM_CLASSES = 60  # 행동 클래스 수 (NTU RGB+D 60)

# --- 학습 하이퍼파라미터 ---
EPOCHS = 100             # 총 학습 에폭
LEARNING_RATE = 0.0003        # 학습률
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 학습 장치
WARMUP_EPOCHS = 3
GRAD_CLIP_NORM = 0.7
T_0 = 10
T_MULT = 2
ETA_MIN = 1e-6
