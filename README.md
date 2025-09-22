# 실험

# 실험1. Shift-GCN + Transformer
`
self.pose_encoder = nn.Sequential(
   nn.Linear(num_joints * 3, 128),
   nn.GELU(),
   nn.Linear(128, 128)
)

self.pos_encoder_64 = PositionalEncoding(d_model=64)
self.pos_encoder_128 = PositionalEncoding(d_model=128)

self.blocks = nn.ModuleList([
    ST_Transformer_Block(in_features=num_coords, out_features=64, num_joints=num_joints),
    ST_Transformer_Block(in_features=64, out_features=128, num_joints=num_joints),
])

self.attention_pool = AttentionPooling(d_model=128)

self.fusion_layer = nn.Sequential(
    nn.Linear(128 * 2, 128),
    nn.GELU(),
    RMSNorm(128)
)

self.dropout = nn.Dropout(p=0.7)
self.fc = nn.Linear(128, num_classes)
`

## 사용한 config.py 설정값

SEED = 42

입력되는 프레임 최대 길이 = 150프레임

배치 크기 = 32

2명 총 관절 수 = 25 * 2

모델 입력 데이터 차원 수 = 7 (거리1 + 방향3 + 가속도3)

분류할 행동 클래스 = 60개

학습률 = 0.0003

에폭 = 100

웜업 에폭 = 3

GRAD_CLIP_NORM = 0.7

CosineAnnealingWarmRestarts 관련 설정값
`
T_0 = 10 # 첫 번째 주기의 길이

T_NULT = 2 # 주기가 반복될수록 길이 T에 곱해줄 값

ETA_MIN = 1e-6 # 최소 학습률
`

## Train.py에서 사용한 것

스케줄러 = WarmUp + CosineAnnealingWarmRestarts

조기 종료(patience 값)는 5

## 데이터 처리

3차원 x, y, z 좌표를 사용하는 대신, 3차원 좌표에서 첫 프레임의 x, y, z 값과 t와 t-1 프레임 사이의 joint 이동 거리, 3D 방향, 가속도를 입력 데이터로 사용합니다.

이러한 데이터 값을 계산하기 전에 Centering을 수행합니다. 모든 관절 좌표에서 joint 0의 좌표를 빼서 절대적인 위치와 상관 없이 자세 정보에만 집중하게 합니다.

변위는 이전 프레임과 현재 프레임 사이의 관절 좌표 차이를 계산해서 벡터를 구합니다.

이동 거리는 변위 벡터의 크기를 L2 Norm으로 계산해서 관절의 속력 정보를 추출합니다.

방향은 변위 벡터를 단위 벡터로 정규화하여 이동 방향 정보를 추출합니다.

변위 벡터들의 시간 변화량을 계산해서 정보를 추출합니다.

그리고 훈련 데이터셋의 모든 특징의 평균과 표준편차(std)를 미리 계산합니다.

모든 행동을 2프레임당 1개씩 샘플링하고, 이를 150 프레임으로 확장합니다. 제일 긴 프레임을 가진 행동 클래스가 300프레임이기 때문입니다.

## 모델의 일반화 성능을 높이기 위해 다음 기법들을 사용했습니다.

임의 회전 50%

가우시안 노이즈 추가 50%

임의 스케일링 50%

관절 마스킹 50%

시간 마스킹 50%

## Z-점수 정규화

데이터 증강이 끝난 후 전처리 단계에서 아까 계산해둔 평균과 표준편차를 사용해 데이터를 정규화합니다.

이를 통해 모든 특징이 비슷한 범위의 값을 가지게 되어 모델의 학습 안정성과 수렴 속도를 향상시킵니다.

# 실험2 Shift-GCN + Mamba
`
self.pose_encoder = nn.Sequential(
    nn.Linear(num_joints * 3, 128),
    nn.GELU(),
    nn.Linear(128, 128)
)

self.blocks = nn.ModuleList([
    ST_Mamba_Block(in_features=num_coords, out_features=64, num_joints=num_joints),
    ST_Mamba_Block(in_features=64, out_features=128, num_joints=num_joints),
])

self.attention_pool = AttentionPooling(d_model=128)

self.fusion_layer = nn.Sequential(
    nn.Linear(128 * 2, 128),
    nn.GELU(),
    RMSNorm(128)
)

self.dropout = nn.Dropout(p=0.7)
self.fc = nn.Linear(128, num_classes)
`

## 사용한 config 설정값

## Train.py에서 사용한 것

스케줄러 = WarmUp + CosineAnnealingWarmRestarts

조기 종료(patience 값)는 5

## 데이터 처리

3차원 x, y, z 좌표를 사용하는 대신, 3차원 좌표에서 첫 프레임의 x, y, z 값과 t와 t-1 프레임 사이의 joint 이동 거리, 3D 방향, 가속도를 입력 데이터로 사용합니다.

이러한 데이터 값을 계산하기 전에 Centering을 수행합니다.

훈련 데이터셋의 모든 특징의 평균과 표준편차(std)를 미리 계산합니다.

모든 행동을 2프레임당 1개씩 샘플링하고, 이를 150 프레임으로 확장합니다.

## 모델의 일반화 성능을 높이기 위해 다음 기법들을 사용했습니다.

임의 회전 50%

가우시안 노이즈 추가 50%

임의 스케일링 50%

관절 마스킹 50%

시간 마스킹 50%

Z-점수 정규화를 진행했습니다.