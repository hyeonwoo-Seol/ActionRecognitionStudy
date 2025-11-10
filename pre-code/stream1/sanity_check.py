# sanity_check.py
#
# 1. 실행 전 config.py 파일에서 규제/증강/드롭아웃을 모두 끄세요.
#    (PROB=1.0, DROPOUT=0.0, ADAMW_WEIGHT_DECAY=0.0, LABEL_SMOOTHING=0.0)
#
# 2. 터미널에서 `python sanity_check.py`를 실행하세요.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import random
import numpy as np

# -----------------------------------------------------
# train.py에서 필요한 모듈과 함수들을 가져옵니다
# -----------------------------------------------------
import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import GCNTransformerModel
# -----------------------------------------------------


# ------------------------------------------------------
# 시드 고정 함수 (train.py에서 복사)
# ------------------------------------------------------
def set_seed(seed):
    random.seed(seed) # 파이썬 내장 random 시드 고정
    np.random.seed(seed) # numpy random 시드 고정
    torch.manual_seed(seed) # Pytorch CPU 연산 시드 고정

    if torch.cuda.is_available(): # GPU 연산 시드 고정
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티-GPU 사용 시
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------
# 과적합 테스트를 수행할 메인 함수
# ------------------------------------------------------
def run_sanity_check():
    # --- 1. 기본 설정 (train.py의 main과 동일) ---
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"--- 🚀 SANTIY CHECK (OVERFITTING TEST) START ---")
    print(f"Using device: {device}")
    print(f"Dataset path: {config.DATASET_PATH}")
    print("WARNING: Make sure PROB=1.0, DROPOUT=0.0, DECAY=0.0, SMOOTHING=0.0 in config.py")

    # --- 2. 데이터 로더 준비 (훈련용만) ---
    train_dataset = NTURGBDDataset(
        data_path = config.DATASET_PATH,
        split = 'train',
        max_frames = config.MAX_FRAMES
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True, # 데이터를 섞어서 한 배치를 뽑음
        num_workers = 0, # 테스트 시에는 0으로 해도 무방
        pin_memory = config.PIN_MEMORY
    )

    # --- 3. 모델, 손실함수, 옵티마이저 준비 ---
    model = GCNTransformerModel(
        layer_dims = config.LAYER_DIMS
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING) # (config.py에서 0.0으로 설정)

    optimizer = optim.AdamW(
        model.parameters(),
        lr = config.LEARNING_RATE,
        weight_decay = config.ADAMW_WEIGHT_DECAY # (config.py에서 0.0으로 설정)
    )

    scaler = GradScaler()

    
    # --- 4. 과적합 테스트 루프 ---
    print("Grabbing one batch from train_loader...")
    try:
        # 훈련 로더에서 딱 한 개의 배치만 가져옵니다.
        fixed_motion_features, fixed_labels = next(iter(train_loader))
    except StopIteration:
        print("Error: train_loader is empty. Check data path/logic.")
        return # 테스트 종료

    # 가져온 배치를 device로 이동시킵니다.
    fixed_motion_features = fixed_motion_features.to(device)
    fixed_labels = fixed_labels.to(device)

    print(f"Batch grabbed. Shape: {fixed_motion_features.shape}, Labels: {fixed_labels.shape}")
    print(f"Starting overfitting test for 300 steps...")

    # 모델을 훈련 모드로 설정
    model.train()

    # 이 배치 *하나만* 300번 반복 학습합니다.
    num_test_steps = 300
    for step in range(num_test_steps):
        optimizer.zero_grad()

        with autocast(device_type=device):
            outputs = model(fixed_motion_features)
            loss = criterion(outputs, fixed_labels)
        
        scaler.scale(loss).backward()
        # (config.py에서 GRAD_CLIP_NORM을 0이 아닌 큰 값(e.g., 1000.0)으로 설정 권장)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        # 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total = fixed_labels.size(0)
        correct = (predicted == fixed_labels).sum().item()
        accuracy = correct / total
        
        if (step + 1) % 10 == 0:
            print(f"Step [{step+1}/{num_test_steps}] | Loss: {loss.item():.6f} | Accuracy: {accuracy:.4f}")

        # 성공 조건 확인
        if loss.item() < 1e-4 and accuracy == 1.0:
            print(f"\n--- ✅ SUCCESS: Overfitting test passed at step {step+1} ---")
            print(f"Loss is near zero ({loss.item():.6f}) and Accuracy is 100%.")
            break
    
    if accuracy != 1.0: # 300 스텝 후에도 100%에 도달 못하면 실패
        print(f"\n--- ❌ FAILURE: Overfitting test failed after {num_test_steps} steps ---")
        print(f"Final Loss: {loss.item():.6f} | Final Accuracy: {accuracy:.4f}")

    print("\n--- SANTIY CHECK COMPLETE ---")


if __name__ == '__main__':
    run_sanity_check()
