# train.py

import torch
# torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import os
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import GCNMambaModel
from utils import calculate_accuracy, save_checkpoint, load_checkpoint

def set_seed(seed):
    # 재현성을 위해 시드를 고정하는 함수
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티-GPU 사용 시
    # cuDNN 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_history(history, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    epochs = range(1, len(history['train_acc']) + 1)

    # Accuracy 플롯 (왼쪽 y축)
    ax1.plot(epochs, history['train_acc'], 'g-', label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'], 'b-', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    # Loss 플롯 (오른쪽 y축)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_loss'], 'r--', label='Train Loss')
    ax2.plot(epochs, history['val_loss'], 'm--', label='Validation Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 그래프 제목 및 범례
    fig.suptitle('Training and Validation History', fontsize=16)
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.savefig(save_path)
    print(f"\nTraining history graph saved to '{save_path}'")
    plt.close()

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    # 한 에폭 동안 모델을 훈련시키는 함수
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    train_bar = tqdm(loader, desc="[Train]", colour="green")
    for data, labels in train_bar:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()

        with autocast(device_type=device): # device_type 인자 추가
            outputs = model(data)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()
                
        running_loss += loss.item() * data.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        train_bar.set_postfix(loss=f"{running_loss/total_samples:.4f}", acc=f"{correct_predictions/total_samples:.4f}")
        
    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc

def validate_one_epoch(model, loader, criterion, device):
    # 한 에폭 동안 모델을 검증하는 함수
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    val_bar = tqdm(loader, desc="[Val]", colour="cyan")
    with torch.no_grad():
        for data, labels in val_bar:
            data, labels = data.to(device), labels.to(device)

            with autocast(device_type=device): # device_type 인자 추가
                outputs = model(data)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            val_bar.set_postfix(loss=f"{running_loss/total_samples:.4f}", acc=f"{correct_predictions/total_samples:.4f}")
            
    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc

def main():
    # 시드 고정
    set_seed(config.SEED)
    print(f"Seed fixed to {config.SEED}")

    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Dataset path: {config.DATASET_PATH}")
    
    # 메인 학습 실행 함수
    # --- 설정값 불러오기 ---
    print(f"Using device: {config.DEVICE}")
    print(f"Dataset path: {config.DATASET_PATH}")
    
    # --- 데이터 로딩 ---
    train_dataset = NTURGBDDataset(
        data_path = config.DATASET_PATH,
        split = 'train',
        max_frames = config.MAX_FRAMES
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKERS,
        pin_memory = config.PIN_MEMORY
    )
    val_dataset = NTURGBDDataset(
        data_path = config.DATASET_PATH,
        split = 'val',
        max_frames = config.MAX_FRAMES
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        pin_memory = config.PIN_MEMORY
    )

    # --- 모델, 손실함수, 옵티마이저 초기화 ---
    model = GCNMambaModel(
        num_joints = config.NUM_JOINTS,
        num_coords = config.NUM_COORDS,
        num_classes = config.NUM_CLASSES
    ).to(device)
    #model = torch.compile(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr = config.LEARNING_RATE,
        weight_decay = 0.01
    )
    scaler = GradScaler()

    
    # 웜업 스케줄러
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor = 0.01,
        end_factor = 1.0,
        total_iters = config.WARMUP_EPOCHS
    )

    # 메인 스케줄러
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max = config.EPOCHS - config.WARMUP_EPOCHS,
        eta_min = 0
    )

    # 두 스케줄러를 순차적으로 연결
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup_scheduler, main_scheduler],
        milestones = [config.WARMUP_EPOCHS]
    )

    # 체크포인트 laod 로직
    start_epoch = 0
    best_accuracy = 0.0
    checkpoint_path = os.path.join(config.SAVE_DIR, "best_model.pth.tar")

    # 조기 종료 변수 추가
    patience = 10
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint '{checkpoint_path}'...")
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, device)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_acc']
        
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            print("GradScaler state loaded.")

        # 스케줄러의 상태도 복원 (start_epoch 만큼 이동)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resuming from epoch {start_epoch}, with best accuracy {best_accuracy:.4f}")
    else:
        print("No checkpoint found, starting training from scratch.")


    try:
        # --- 전체 에폭 학습 루프 ---
        for epoch in range(start_epoch, config.EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            

            # 매 epoch 결과 기록
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            print(f"Epoch {epoch+1} Summary | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            # 최고 검증 정확도를 가진 모델 저장
            if val_acc > best_accuracy:
                print(f"New best accuracy: {val_acc:.4f}! Saving model...")
                best_accuracy = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_accuracy,
                    'scaler': scaler.state_dict()
                }, directory=config.SAVE_DIR, filename="best_model.pth.tar")
                patience_counter = 0 # 최고 기록 경신 시 카운터 초기화
            else:
                patience_counter += 1
                print(f"No imporvement in validation accuracy for {patience_counter} epoch(s).")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    except KeyboardInterrupt:
        # 사용자가 Ctrl+C를 눌렀을 때 실행되는 부분
        print("\n\nUser interrupted training. Generating graph with current history...")
            
    # 학습이 정상적으로 모두 끝나거나, KeyboardInterrupt로 중단되었을 때 이 코드가 실행됩니다.
    if history['train_acc']: # 기록이 한 번이라도 되었으면 그래프 생성
        plot_history(history, "training_history.png")

    # 최종적으로 가장 좋았던 모델의 성능을 출력합니다.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        best_val_acc = checkpoint['best_acc']
        print(f"\nBest Validation Accuracy achieved: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    print("\nTraining finished.")
    
if __name__ == '__main__':
    main()
