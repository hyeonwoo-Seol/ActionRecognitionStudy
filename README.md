# 연구 주제

## 연구 방법론

```문제 해결을 위해 어떤 모델, 알고리즘, 데이터셋을 사용했는지 구체적으로 기술해야 합니다. 예를 들어, '이미지 분류 정확도를 높이기 위해 ResNet50 모델을 사용하여 CIFAR-100 데이터셋으로 학습을 진행했다'와 같이 명시적인 정보가 중요합니다. (이 내용은 지우기)
```

## 실험 최종 결과

## 실험1

```
# No nomalize by bone length in preprocess_ntu_data.py
# python train.py --scheduler cosine_decay

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
LAYER_DIMS = [64, 128]
USE_GCN = True

```

24 epoch에서 학습률 0.00027 -> 0.00002

### 결과1

![Study1](image/Study1.png)

Test Acc: 64.7%

## 실험2

```
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
```

### 결과2

## 실험3

### 결과3

## 실험4

### 결과4

## 실험5

### 결과5
