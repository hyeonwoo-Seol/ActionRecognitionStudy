# train.py

# ## --------------------------------------------------------------------------
# model.py의 모델을 사용하여 NTU-RGB+D 데이터셋을 학습하고
# 검증하는 전체 파이프라인을 포함한다.
# 주요 기능:
# 1. 재현성을 위한 시드 고정
# 2. 데이터 로딩 및 전처리
# 3. 모델, 손실 함수, 옵티마이저, 학습률 스케줄러 초기화
# 4. 학습 및 검증 루프 실행
# 5. 체크포인트 저장 및 불러오기
# 6. 조기 종료 
# 7. 학습 과정 시각화 및 저장
# ## --------------------------------------------------------------------------


import torch
# torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from tqdm import tqdm
import os
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import GCNTransformerModel
from utils import calculate_accuracy, save_checkpoint, load_checkpoint




# ## ------------------------------------------------------
# 시드 고정 함수
# ## ------------------------------------------------------
def set_seed(seed):
    random.seed(seed) # 파이썬 내장 random 시드 고정
    np.random.seed(seed) # numpy random 시드 고정
    torch.manual_seed(seed) # Pytorch CPU 연산 시드 고정

    if torch.cuda.is_available(): # GPU 연산 시드 고정
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티-GPU 사용 시
    
    # >> cuDNN 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # False -> 동일한 입력 크기에 대해 같은 알고리즘 적용




# ## ---------------------------------------------------------
# 학습 및 검증 과정의 Loss와 Acc를 기록하고 그래프로 만들기
# ## ---------------------------------------------------------
def plot_history(history, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # >> Epoch 수를 X축으로 설정
    epochs = range(1, len(history['train_acc']) + 1)

    # >> 왼쪽 Y축에 정확도
    ax1.plot(epochs, history['train_acc'], 'g-', label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'], 'b-', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    # >> 오른쪽 Y축에 손실
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_loss'], 'r--', label='Train Loss')
    ax2.plot(epochs, history['val_loss'], 'm--', label='Validation Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # >> 그래프 제목 및 범례
    fig.suptitle('Training and Validation History', fontsize=16)
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    # >> 완성된 그래프를 이미지로 저장한다.
    plt.savefig(save_path)
    print(f"\nTraining history graph saved to '{save_path}'")
    plt.close()




# ## -----------------------------------------------------------------
# epcoh 동안 모델의 학습을 수행한다.
# ## -----------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    # >> 모델을 학습 모드로 설정한다.
    model.train()

    # >> 모든 Epoch들의 총 손실, 맞춘 예측 수, 전체 샘플 수를 기록할 변수들이다.
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    

    train_bar = tqdm(loader, desc="[Train]", colour="green")
    # >> 데이터 로더로부터 미니배치를 받아서 학습을 진행한다.
    for motion_features, labels, first_frame_coords in train_bar:
        # >> 모션 특징, 레이블, 첫 프레임 좌표값을 지정된 device로 이동시킨다.
        motion_features = motion_features.to(device)
        labels = labels.to(device)
        first_frame_coords = first_frame_coords.to(device)
        
        # >> 이전 배치의 기울기를 초기화한다.
        # >> 이를 하지 않으면 이전 배치들의 오차에 현재 배치가 영향을 받는다.
        optimizer.zero_grad()


        # >> autocast Context 내에서 연산을 수행하여 자동 혼합 정밀도인 AMP를 사용한다.
        # >> 성능 향상 및 메모리 사용량 감소를 위함이다.
        with autocast(device_type=device):
            # >> 순전파를 수행하여 예측 결과를 얻는다.
            outputs = model(motion_features, first_frame_coords)
            # >> 손실을 계산한다.
            loss = criterion(outputs, labels)
        
        # ## ------------------------------------------------------------------------------
        # AMP로 float32 대신 float16으로 연산을 수행하면 역전파 과정에서 기울기값이
        # 0이 되는 현상이 발생할 수 있다. 이를 해결하기 위해 GradScaler를 사용한다.
        # loss에 큰 수를 곱한 뒤 역전파를 하고, 그 다음에 원상 복구 후 기울기 클리핑을 적용한다.
        # 원상 복구하는 이유는 기울기 클리핑을 정확하게 수행하고 가중치를 올바르게 업데이트하기 위함이다.
        # ## -------------------------------------------------------------------------------
        # >> GradScaler를 사용하여 손실 값 스케일을 조정하고 역전파한다.
        scaler.scale(loss).backward()
        # >> 옵티마이저가 Step을 진행하기 전에, 스케일된 기울기를 복원한다.
        scaler.unscale_(optimizer)
        # >> 기울기 폭발을 방지하기 위해 기울기 클리핑을 수행한다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        # >> 옵티마이저를 사용하여 가중치를 업데이트한다.
        scaler.step(optimizer)
        # >> 다음 반복을 위해 GradScaler의 스케일 팩터를 업데이트한다.
        scaler.update()


        # >> 배치 손실, 정확도, 샘플 수를 누적한다.
        running_loss += loss.item() * motion_features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # >> 진행률 표시줄에 현재 손실과 정확도를 표시한다.
        train_bar.set_postfix(loss=f"{running_loss/total_samples:.4f}", acc=f"{correct_predictions/total_samples:.4f}")

    # >> 에폭의 평균 손실과 정확도를 반환한다.
    return running_loss / total_samples, correct_predictions / total_samples




# ## -----------------------------------------------------------------
# epcoh 동안 모델의 검증을 수행한다.
# ## -----------------------------------------------------------------
def validate_one_epoch(model, loader, criterion, device):
    # >> 모델을 평가 모드로 설정한다.
    model.eval()

    # >> 모든 Epoch들의 총 손실, 맞춘 예측 수, 전체 샘플 수를 기록할 변수들이다.
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    val_bar = tqdm(loader, desc="[Val]", colour="cyan")

    # >> 기울기 계산을 비활성화하여 검증 속도를 높이고 메모리 사용량을 줄인다.
    with torch.no_grad():
        # >> 데이터 로더로부터 미니배치를 받아서 검증을 진행한다.
        for motion_features, labels, first_frame_coords in val_bar:
            # >> 모션 특징, 레이블, 첫 프레임 좌표값을 지정된 device로 이동시킨다.
            motion_features = motion_features.to(device)
            labels = labels.to(device)
            first_frame_coords = first_frame_coords.to(device)


            # >> autocast를 사용하여 혼합 정밀도로 순전파를 수행한다.
            with autocast(device_type=device):
                # >> 순전파를 수행하여 예측 결과를 얻는다.
                outputs = model(motion_features, first_frame_coords)
                # >> 손실을 계산한다.
                loss = criterion(outputs, labels)
            

            # >> 배치 손실, 정확도, 샘플 수를 누적한다.
            running_loss += loss.item() * motion_features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            val_bar.set_postfix(loss=f"{running_loss/total_samples:.4f}", acc=f"{correct_predictions/total_samples:.4f}")
    
    # >> 에폭의 평균 손실과 정확도를 반환한다.
    return running_loss / total_samples, correct_predictions / total_samples


def main():
    # >> 재현성을 위해 시드를 고정한다.
    set_seed(config.SEED)
    print(f"Seed fixed to {config.SEED}")


    # >> 학습에 사용할 장치(CPU 또는 GPU)를 설정한다.
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Dataset path: {config.DATASET_PATH}")
    
    
    # >> 설정값 불러오기
    print(f"Using device: {config.DEVICE}")
    print(f"Dataset path: {config.DATASET_PATH}")
    

    # >> 학습용 데이터셋을 초기화한다.
    train_dataset = NTURGBDDataset(
        data_path = config.DATASET_PATH,
        split = 'train',
        max_frames = config.MAX_FRAMES
    )

    # >> 학습용 데이터 로더를 설정한다.
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKERS,
        pin_memory = config.PIN_MEMORY
    )

    # >> 검증용 데이터셋을 초기화한다.
    val_dataset = NTURGBDDataset(
        data_path = config.DATASET_PATH,
        split = 'val',
        max_frames = config.MAX_FRAMES
    )

    # >> 검증용 데이터 로더를 설정한다.
    val_loader = DataLoader(
        val_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        pin_memory = config.PIN_MEMORY
    )

    # >> GCN-Transformer 모델을 초기화하고 지정된 장치로 이동시킨다.
    model = GCNTransformerModel(
        num_joints = config.NUM_JOINTS,
        num_coords = config.NUM_COORDS,
        num_classes = config.NUM_CLASSES
    ).to(device)
    

    # >> 손실 함수로 CrossEntropyLoss를 사용하며, 레이블 스무딩을 적용한다.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # >> 옵티마이저로 AdamW를 사용하여 가중치 감쇠를 적용한다.
    optimizer = optim.AdamW(
        model.parameters(),
        lr = config.LEARNING_RATE,
        weight_decay = 0.01
    )

    # >> 자동 혼합 정밀도(AMP) 학습을 위한 GradScaler를 초기화한다.
    scaler = GradScaler()

    # >> 스케줄러는 변경 예정 ----------------------------------------------------------
    # >> 1. 웜업 스케줄러 (Warmup Scheduler)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor = 0.01,
        end_factor = 1.0,
        total_iters = config.WARMUP_EPOCHS
    )

    # >> 2. 메인 스케줄러 (CosineAnnealingWarmRestarts)
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = config.T_0,          # 첫 번째 주기의 길이
        T_mult = config.T_MULT,    # 주기가 반복될수록 길이 T_i에 곱해줄 값
        eta_min = config.ETA_MIN   # 최소 학습률
    )

    # >> 3. 두 스케줄러를 순차적으로 연결
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup_scheduler, main_scheduler],
        milestones = [config.WARMUP_EPOCHS]
    )


    # >> 학습을 시작할 에폭, 최고 정확도 등 상태 변수를 초기화한다.
    start_epoch = 0
    best_accuracy = 0.0
    last_val_acc = 0.0


    # >> 체크포인트 파일이 저장될 경로를 설정한다.
    checkpoint_path = os.path.join(config.SAVE_DIR, "best_model.pth.tar")


    # >> 조기 종료 변수를 추가한다.
    patience = 10
    patience_counter = 0


    # >> 학습 및 검증 과정의 손실과 정확도를 기록할 딕셔너리를 초기화한다.
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


    # >> 저장된 체크포인트 파일이 있으면 학습을 이어서 진행한다.
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint '{checkpoint_path}'...")

        # >> 체크포인트 파일을 불러와 모델과 옵티마이저의 상태를 복원한다.
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, device)

        # >> 마지막으로 저장된 에폭 다음부터 학습을 시작한다.
        start_epoch = checkpoint['epoch']

        # >> 이전 학습에서 기록된 최고 정확도를 불러온다.
        best_accuracy = checkpoint['best_acc']
        last_val_acc = checkpoint.get('last_val_acc', 0.0)
        

        # >> GradScaler의 상태도 복원한다.
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            print("GradScaler state loaded.")

        # >> 현재 시작 에폭에 맞게 스케줄러의 상태를 업데이트한다.
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resuming from epoch {start_epoch}, with best accuracy {best_accuracy:.4f}")
    else: # >> 체크포인트 파일이 없으면 처음부터 학습을 시작한다.
        print("No checkpoint found, starting training from scratch.")


    try:
        # >> 전체 에폭 학습 루프
        for epoch in range(start_epoch, config.EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
            
            # >> 한 에폭 동안 모델을 학습시킨다.
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            
            # >> 한 에폭 동안 모델을 검증한다.
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            

            # >> 현재 에폭의 학습 및 검증 결과를 history 딕셔너리에 추가한다.
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # >> 학습률 스케줄러를 다음 단계로 업데이트한다.
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            print(f"Epoch {epoch+1} Summary | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            
            
            # >> 최고 검증 정확도를 가진 모델을 저장한다.
            if val_acc > best_accuracy:
                print(f"New best accuracy: {val_acc:.4f}! Saving model...")
                best_accuracy = val_acc
                # >> 현재 모델의 상태를 체크포인트 파일로 저장한다.
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_accuracy,
                    'scaler': scaler.state_dict(),
                    'last_val_acc': last_val_acc
                }, directory=config.SAVE_DIR, filename="best_model.pth.tar")
                patience_counter = 0 # 최고 기록 경신 시 카운터 초기화
            else:
                patience_counter += 1
                print(f"No imporvement in validation accuracy for {patience_counter} epoch(s).")


            # >> 조기 종료 카운터가 설정된 patience 값에 도달하면
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
    
    # >> 사용자가 Ctrl+C를 눌러 학습을 중단했을 때 실행된다.
    except KeyboardInterrupt:
        print("\n\nUser interrupted training. Generating graph with current history...")


           
    # >> 학습이 정상적으로 모두 끝나거나, KeyboardInterrupt로 중단되었을 때 이 코드가 실행된다.
    if history['train_acc']: # 기록이 한 번이라도 되었으면 그래프 생성
        plot_history(history, "training_history.png")


    # >> 최종적으로 가장 좋았던 모델의 성능을 출력한다.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        best_val_acc = checkpoint['best_acc']
        print(f"\nBest Validation Accuracy achieved: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    print("\nTraining finished.")


if __name__ == '__main__':
    main()