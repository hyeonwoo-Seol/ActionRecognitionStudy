# config.py

import torch


# >> NTU RGB+D 60 데이터셋의 .np 파일들이 있는 디렉토리 경로
# >> 이 경로에는 preprocess_ntu_data.py를 실행하고 생성된 .np 파일들이 있는 경로로 지정해야 한다.
DATASET_PATH = '../nturgbd_processed_allNew/'

# >> 학습된 모델 가중치(체크포인트)를 저장할 디렉토리
SAVE_DIR = 'checkpoints/'

# >> 재현성을 위해 시드 설정
SEED = 42

# >> 데이터 로더 설정
MAX_FRAMES = 150  # 시퀀스의 최대 길이 (패딩 또는 절단 기준)
BATCH_SIZE = 38   # 배치 크기
NUM_WORKERS = 6   # 데이터를 불러올 때 사용할 CPU 프로세서 수 
PIN_MEMORY = True # GPU 사용 시 데이터 전송 속도를 높이기 위한 설정

# >> 모델 하이퍼파라미터
NUM_JOINTS = 50   # 관절 수
NUM_COORDS = 15   # 거리(1) + 방향(3) + 뼈길이(1) + 관절각도(1) + 몸통상대각도(2) + 비인접관절거리(2) + P0-P1중심거리(1) + P0-P1손발거리(4)
NUM_CLASSES = 60  # 행동 클래스 수 (NTU RGB+D 60)
PROB = 0.6 # -> 데이터 증강 확률. 60% 확률로 증강을 '안'한다.
SPATIAL_KEEP_RATE = 0.7 # 공간적 어텐션에서 '중요 관절'로 유지할 비율
NUM_SUBJECTS = 2

# >> 학습 하이퍼파라미터
EPOCHS = 100             # 총 학습 에폭
LEARNING_RATE = 0.0003   # 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 학습 장치
WARMUP_EPOCHS = 10        # 학습 초기에 학습률을 서서히 증가시키는 웜업 에폭 수 
GRAD_CLIP_NORM = 1.0     # 그레이디언트 폭발을 막기 위한 클리핑 최대 L2 Norm 값 
ADAMW_WEIGHT_DECAY = 0.05 # 이전값: 0.01 AdamW weight decay , L2 정규화의 강도 설정
PATIENCE = 10 # 조기종료 변수
LABEL_SMOOTHING = 0.05 # 이전값: 0.05 Loss Function CrossEntropy의 label smoothing
DROPOUT = 0.3 # 이전값: 0.2 dropout
ETA_MIN = 1e-6
T_0 = 100 # n에폭마다 학습률이 최대치로 초기화된다. 현재는 사용하지 않기 위해 100으로 설정했다.
T_MULT = 1 # 2로 설정된 경우, 15에폭 다음에 학습률 초기화는 2배인 30에폭이다. 현재는 사용하지 않는다.
ADVERSARIAL_ALPHA = 0.1 # 적대적 학습의 강도

# >> DIMS 사용법은, [입력 크기, 출력 크기]이다. 만약 4층을 쌓으려면 [64, 128, 128, 256, 256] 하면 된다.
FAST_DIMS = [32, 32, 64]      # Fast: 가볍게
SLOW_DIMS = [128, 128, 256]    # Slow: 무겁게
