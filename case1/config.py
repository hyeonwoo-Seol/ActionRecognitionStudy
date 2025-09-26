# config.py

import torch


# >> NTU RGB+D 60 데이터셋의 .np 파일들이 있는 디렉토리 경로
# >> 이 경로에는 preprocess_ntu_data.py를 실행하고 생성된 .np 파일들이 있는 경로로 지정해야 한다.
DATASET_PATH = '../nturgbd_processed/'

# >> 학습된 모델 가중치(체크포인트)를 저장할 디렉토리
SAVE_DIR = 'checkpoints/'

# >> 재현성을 위해 시드 설정
SEED = 42

# >> 데이터 로더 설정
MAX_FRAMES = 150  # 시퀀스의 최대 길이 (패딩 또는 절단 기준)
BATCH_SIZE = 32   # 배치 크기
NUM_WORKERS = 6   # 데이터를 불러올 때 사용할 CPU 프로세서 수 
PIN_MEMORY = True # GPU 사용 시 데이터 전송 속도를 높이기 위한 설정

# >> 모델 하이퍼파라미터
NUM_JOINTS = 50   # 관절 수
NUM_COORDS = 7    # 거리1 + 방향3 + 가속도3
NUM_CLASSES = 60  # 행동 클래스 수 (NTU RGB+D 60)
PROB = 0.5 # 데이터 증강 확률

# >> 학습 하이퍼파라미터
EPOCHS = 100             # 총 학습 에폭
LEARNING_RATE = 0.0003   # 학습률
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 학습 장치
WARMUP_EPOCHS = 3        # 학습 초기에 학습률을 서서히 증가시키는 웜업 에폭 수 
GRAD_CLIP_NORM = 0.7     # 그레이디언트 폭발을 막기 위한 클리핑 최대 L2 Norm 값 
ADAMW_WEIGHT_DECAY = 0.1 # AdamW weight decay , L2 정규화의 강도 설정
PATIENCE = 10 # 조기종료 변수
LABEL_SMOOTHING = 0.1 # Loss Function CrossEntropy의 label smoothing
DROPOUT = 0.5 # dropout
ETA_MIN = 1e-6
T_0 = 15 # 15에폭마다 학습률이 최대치로 초기화된다.
T_MULT = 2 # 15 에폭 다음 학습률 초기화는 그 2배인 30에폭이다.

# >> block_type, layer_dims, use_gcn를 설정한다.
# >> block_type='st' : Shift-GCN + ST-Transformer, block_type='standard' : Shift-GCN + Transformer
# >> gcn 사용 여부는 use_gcn을 True, False로 지정한다.
# >> layer_dims 사용법은, [입력 크기, 출력 크기]이다. 만약 4층을 쌓으려면 [64, 128, 128, 256, 256] 하면 된다.
BLOCK_TYPE = 'st'
LAYER_DIMS = [64, 128, 256]
USE_GCN = True
