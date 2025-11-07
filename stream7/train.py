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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR
from tqdm import tqdm
import os
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import ConcatDataset
import time
import optuna

import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import SlowFast_GCNTransformer
from utils import calculate_accuracy, save_checkpoint, load_checkpoint

# >> --study-name " "은 모든 실험 기록을 저장할 데이터베이스 파일(.db) 이름이다.
# python train.py --protocol xsub --scheduler cosine_decay --study-name "slowfast_tuning_v1" --n-trials 50
# >> 실시간 모니터링을 하려면 새 터미널에서 optuna-dashboard sqlite:///slowfast_tuning_v1.db 를 입력하라.

# python train.py --scheduler cosine_decay or cosine_restarts
# python train.py --protocol xsub or xview

# ## ----------------------------------------------------
# 커맨드 라인 인자에 따라 스케줄러를 반환한다.
# 사용 방법은 다음과 같다.
# python train.py 는 cosine_decay가 기본값이다.
# python train.py --scheduler cosine_decay
# python train.py --scheduler cosine_restarts
# ## ----------------------------------------------------



# Optuna가 한 번의 Trial(시도)마다 학습할 최대 에폭 수
# (값을 줄여서 더 많은 하이퍼파라미터를 빠르게 탐색하는 것이 유리합니다)
MAX_EPOCHS_PER_TRIAL = 30 
# Optuna가 총 시도할 횟수
N_TRIALS = 50 
# Optuna가 가지치기(Pruning)를 시작하기 전, 최소한으로 학습을 보장할 에폭 수
PRUNING_WARMUP_EPOCHS = 10

def get_scheduler(optimizer, scheduler_name, total_epochs, warmup_epochs):
    print(f"Using '{scheduler_name}' scheduler.")

    # >> 1. 웜업 스케줄러는 공통으로 사용한다.
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    
    # >> 2. 선택된 이름에 따라 메인 스케줄러를 설정한다.
    if scheduler_name == 'cosine_decay':
        # >> 옵션 1: 워밍업 + 재시작 없는 코사인 감쇠 (CosineAnnealingLR)
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=config.ETA_MIN
        )
    elif scheduler_name == 'cosine_restarts':
        # >> 옵션 2: 워밍업 + 재시작 있는 코사인 감쇠 (CosineAnnealingWarmRestarts)
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_MULT,
            eta_min=config.ETA_MIN
        )
    else:
        # >> 잘못된 이름이 입력되면 에러를 발생시킨다.
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # >> 3. 두 스케줄러를 순차적으로 연결하여 최종 스케줄러를 반환한다.
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )





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
def train_one_epoch(model, loader, criterion_action, criterion_subject, optimizer, device, scaler):
    # >> 모델을 학습 모드로 설정한다.
    model.train()

    # >> 모든 Epoch들의 총 손실, 맞춘 예측 수, 전체 샘플 수를 기록할 변수들이다.
    running_loss = 0.0
    correct_action = 0
    correct_subject = 0
    total_samples = 0

    

    train_bar = tqdm(loader, desc="[Train]", colour="green", leave=False)
    # >> 데이터 로더로부터 미니배치를 받아서 학습을 진행한다.
    for data_fast, data_slow, action_labels, subject_labels in train_bar:
        # >> 두 개의 데이터 텐서를 device로 이동시킨다.
        data_fast = data_fast.to(device)
        data_slow = data_slow.to(device)
        action_labels = action_labels.to(device)
        subject_labels = subject_labels.to(device)
        
        
        
        # >> 이전 배치의 기울기를 초기화한다.
        # >> 이를 하지 않으면 이전 배치들의 오차에 현재 배치가 영향을 받는다.
        optimizer.zero_grad()


        # >> autocast Context 내에서 연산을 수행하여 자동 혼합 정밀도인 AMP를 사용한다.
        # >> 성능 향상 및 메모리 사용량 감소를 위함이다.
        with autocast(device_type=device):
            # >> 순전파를 수행하여 예측 결과를 얻는다.
            outputs_action, outputs_subject = model(data_fast, data_slow)

            # --- [수정됨] 정보 누수 방지 로직 ---

            # 1. Action 손실을 샘플별로 계산 (reduction='none'이므로) (N,)
            loss_action_all = criterion_action(outputs_action, action_labels)
            
            # 2. Subject 손실은 배치 전체에 대해 계산 (reduction='mean')
            loss_subject = criterion_subject(outputs_subject, subject_labels)
            
            # 3. Source 데이터(subject_label == 0)에만 Action 손실을 적용
            source_mask = (subject_labels == 0).float()

            # 4. 마스크가 적용된 Action 손실의 평균을 계산
            # (마스크된 샘플 수로만 나누어 평균을 구함)
            loss_action = (loss_action_all * source_mask).sum() / (source_mask.sum() + 1e-8)

            
            # GRL을 사용하므로, 두 손실을 config의 alpha 값으로 더함
            loss = loss_action + (config.ADVERSARIAL_ALPHA * loss_subject)

            
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
        running_loss += loss.item() * data_fast.size(0)
        total_samples += action_labels.size(0)
        
        _, predicted_action = torch.max(outputs_action.data, 1)
        correct_action += (predicted_action == action_labels).sum().item()
        
        _, predicted_subject = torch.max(outputs_subject.data, 1)
        correct_subject += (predicted_subject == subject_labels).sum().item()
        
        train_bar.set_postfix(
            loss=f"{running_loss/total_samples:.4f}",
            acc_ACT=f"{correct_action/total_samples:.4f}", # 행동 정확도
            acc_SUB=f"{correct_subject/total_samples:.4f}"  # 피실험자 정확도
        )
        
    return (running_loss / total_samples, correct_action / total_samples, correct_subject / total_samples)




# ## -----------------------------------------------------------------
# epcoh 동안 모델의 검증을 수행한다.
# ## -----------------------------------------------------------------
def validate_one_epoch(model, loader, criterion_action, criterion_subject, device):
    # >> 모델을 평가 모드로 설정한다.
    model.eval()

    # >> 모든 Epoch들의 총 손실, 맞춘 예측 수, 전체 샘플 수를 기록할 변수들이다.
    running_loss = 0.0
    correct_action = 0
    correct_subject = 0
    total_samples = 0
    
    val_bar = tqdm(loader, desc="[Val]", colour="cyan", leave=False)

    # >> 기울기 계산을 비활성화하여 검증 속도를 높이고 메모리 사용량을 줄인다.
    with torch.no_grad():
        # >> 데이터 로더로부터 미니배치를 받아서 검증을 진행한다.
        for data_fast, data_slow, action_labels, subject_labels in val_bar:
            # >> 두 개의 데이터 텐서를 device로 이동시킨다.
            data_fast = data_fast.to(device)
            data_slow = data_slow.to(device)
            action_labels = action_labels.to(device)
            subject_labels = subject_labels.to(device)


            # >> autocast를 사용하여 혼합 정밀도로 순전파를 수행한다.
            with autocast(device_type=device):
                # >> 순전파를 수행하여 예측 결과를 얻는다.
                outputs_action, outputs_subject = model(data_fast, data_slow)
                # >> 손실을 계산한다.
                loss_action_all = criterion_action(outputs_action, action_labels)

                # [추가] 검증 시에는 텐서의 평균값을 손실로 사용
                loss_action = loss_action_all.mean() 

                loss_subject = criterion_subject(outputs_subject, subject_labels)


                loss = loss_action
                

            # >> 배치 손실, 정확도, 샘플 수를 누적한다.
            running_loss += loss.item() * data_fast.size(0)
            total_samples += action_labels.size(0)
            
            _, predicted_action = torch.max(outputs_action.data, 1)
            correct_action += (predicted_action == action_labels).sum().item()
            
            _, predicted_subject = torch.max(outputs_subject.data, 1)
            correct_subject += (predicted_subject == subject_labels).sum().item()
            
            val_bar.set_postfix(
                loss=f"{running_loss/total_samples:.4f}",
                acc_ACT=f"{correct_action/total_samples:.4f}",
                acc_SUB=f"{correct_subject/total_samples:.4f}"
            )
    
    return (running_loss / total_samples, 
            correct_action / total_samples, 
            correct_subject / total_samples)

# --- [OPTUNA] ---
# 기존 main() 함수의 내용을 Optuna가 호출할 'objective' 함수(run_trial)로 변경합니다.
# --- [OPTUNA] ---
def run_trial(trial, args):
    """Optuna가 단일 trial을 실행하기 위해 호출하는 함수"""
    
    # >> 재현성을 위해 시드를 고정한다. (Optuna는 trial마다 다른 시드를 권장할 수 있으나,
    #    하이퍼파라미터의 순수 효과를 보려면 시드 고정이 유리할 수 있습니다.)
    set_seed(config.SEED)

    # --- [OPTUNA] 1. 하이퍼파라미터 제안 ---
    # Optuna의 trial 객체를 사용하여 config.py의 값들을 '임시로' 덮어씁니다.
    config.LEARNING_RATE = trial.suggest_float("LEARNING_RATE", 1e-4, 5e-4, log=True)
    config.DROPOUT = trial.suggest_float("DROPOUT", 0.2, 0.5)
    config.ADVERSARIAL_ALPHA = trial.suggest_float("ADVERSARIAL_ALPHA", 0.05, 0.3)
    config.PROB = trial.suggest_float("PROB", 0.3, 0.7) # (증강 안 할 확률: 30% ~ 70%)
    config.ADAMW_WEIGHT_DECAY = trial.suggest_float("ADAMW_WEIGHT_DECAY", 0.01, 0.1, log=True)
    config.LABEL_SMOOTHING = trial.suggest_float("LABEL_SMOOTHING", 0.0, 0.15)
    
    # (config.py에 있지만 이번 튜닝에서 제외할 파라미터는 기존 값을 사용합니다)
    
    print(f"\n--- [Trial {trial.number}] ---")
    print(f"Params: LR={config.LEARNING_RATE:.6f}, Dropout={config.DROPOUT:.3f}, Alpha={config.ADVERSARIAL_ALPHA:.3f}")
    print(f"        Prob={config.PROB:.3f}, WeightDecay={config.ADAMW_WEIGHT_DECAY:.4f}, Smoothing={config.LABEL_SMOOTHING:.3f}")

    # --- [OPTUNA] 2. 학습 준비 (기존 main() 코드) ---
    device = config.DEVICE

    # (데이터 로더는 Optuna trial마다 새로 생성해야 합니다.
    #  특히 ntu_data_loader.py가 config.PROB 값을 참조하기 때문입니다)
    train_dataset_source = NTURGBDDataset(
        data_path = config.DATASET_PATH, split = 'train',
        max_frames = config.MAX_FRAMES, protocol = args.protocol
    )
    train_dataset_target = NTURGBDDataset(
        data_path = config.DATASET_PATH, split = 'val',
        max_frames = config.MAX_FRAMES, protocol = args.protocol
    )
    combined_train_dataset = ConcatDataset([train_dataset_source, train_dataset_target])
    train_loader = DataLoader(
        combined_train_dataset, batch_size = config.BATCH_SIZE, shuffle = True,
        num_workers = config.NUM_WORKERS, pin_memory = config.PIN_MEMORY
    )
    val_dataset = NTURGBDDataset(
        data_path = config.DATASET_PATH, split = 'val',
        max_frames = config.MAX_FRAMES, protocol = args.protocol
    )
    val_loader = DataLoader(
        val_dataset, batch_size = config.BATCH_SIZE, shuffle = False,
        num_workers = config.NUM_WORKERS, pin_memory = config.PIN_MEMORY
    )

    # (덮어쓴 config 값으로 모델 생성)
    model = SlowFast_GCNTransformer(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        fast_dims=config.FAST_DIMS,
        slow_dims=config.SLOW_DIMS
    ).to(device)

    # (덮어쓴 config 값으로 손실 함수, 옵티마이저 생성)
    criterion_action = nn.CrossEntropyLoss(
        label_smoothing=config.LABEL_SMOOTHING, reduction='none'
    )
    criterion_subject = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr = config.LEARNING_RATE,
        weight_decay = config.ADAMW_WEIGHT_DECAY
    )
    scaler = GradScaler()
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        total_epochs=MAX_EPOCHS_PER_TRIAL, # --- [OPTUNA] ---
        warmup_epochs=config.WARMUP_EPOCHS
    )

    best_accuracy = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- [OPTUNA] --- 각 trial마다 고유한 체크포인트 폴더 생성
    trial_save_dir = os.path.join(config.SAVE_DIR, f"trial_{trial.number}")
    os.makedirs(trial_save_dir, exist_ok=True)
    
    # --- [OPTUNA] --- Trial은 항상 0 에폭부터 새로 시작 (체크포인트 로드 로직 제거)
    
    LR_DROP_PATIENCE = 4
    MAX_LR_DROPS = 2
    patience_counter = 0
    lr_drop_count = 0

    try:
        # --- [OPTUNA] --- config.EPOCHS 대신 MAX_EPOCHS_PER_TRIAL 사용
        for epoch in range(MAX_EPOCHS_PER_TRIAL):
            epoch_start_time = time.time()
            print(f"\n--- [Trial {trial.number}] Epoch {epoch+1}/{MAX_EPOCHS_PER_TRIAL} ---")
            
            train_loss, train_acc_action, train_acc_subject = train_one_epoch(
                model, train_loader, criterion_action, criterion_subject, optimizer, device, scaler
            )
            
            val_loss, val_acc_action, val_acc_subject = validate_one_epoch(
                model, val_loader, criterion_action, criterion_subject, device
            )

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc_action)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc_action)
            
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} Summary | Train Acc: {train_acc_action:.4f} | Val Acc: {val_acc_action:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            print(f"(Adversarial) | Train Sub_Acc: {train_acc_subject:.4f} | Val Sub_Acc: {val_acc_subject:.4f}")

            if val_acc_action > best_accuracy:
                best_accuracy = val_acc_action
                patience_counter = 0 
                print(f"New best accuracy for Trial {trial.number}: {val_acc_action:.4f}! Saving model...")

                # --- [OPTUNA] --- 고유한 경로에 저장
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': best_accuracy,
                    'scaler': scaler.state_dict(),
                    'history': history,
                    'patience_counter': patience_counter,
                    'lr_drop_count': lr_drop_count,
                    'optuna_params': trial.params # --- [OPTUNA] --- 이 모델을 만든 파라미터 저장
                }, directory=trial_save_dir, filename="best_model.pth.tar")

            else: 
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{LR_DROP_PATIENCE}. LR Drops: {lr_drop_count}/{MAX_LR_DROPS}.")
                
                if patience_counter >= config.PATIENCE: 
                    print(f"Early stopping triggered for Trial {trial.number} (Patience: {config.PATIENCE}).")
                    break # 이 Trial의 학습 루프 조기 종료
                

            
            # --- [OPTUNA] 3. Pruning (가지치기) ---
            # Optuna에 현재 epoch의 Val Acc를 보고합니다.
            trial.report(val_acc_action, epoch)

            # Optuna가 이 trial을 중단해야 한다고 결정했는지 확인합니다.
            if epoch > PRUNING_WARMUP_EPOCHS and trial.should_prune():
                print(f"--- [Trial {trial.number}] Pruned at epoch {epoch+1} (Val Acc: {val_acc_action:.4f}) ---")
                # 가지치기 예외를 발생시켜 이 trial을 즉시 중단합니다.
                raise optuna.TrialPruned()

    except KeyboardInterrupt:
        # --- [OPTUNA] --- 사용자가 Ctrl+C를 눌러도 이 trial만 중단되고,
        # Optuna의 메인 프로세스(study.optimize)는 계속 다음 trial을 시도할 수 있습니다.
        # (전체 연구를 중단하려면 메인 프로세스에서 Ctrl+C를 눌러야 함)
        print(f"\nUser interrupted Trial {trial.number}.")
        # 이 trial은 실패한 것으로 간주하고 최저값을 반환할 수 있습니다.
        return 0.0 # 또는 best_accuracy
    
    # --- [OPTUNA] ---
    # 그래프를 trial마다 저장합니다.
    plot_history(history, os.path.join(trial_save_dir, "training_history.png"))
    
    # 이 trial의 최종 '최고 정확도'를 반환합니다.
    # Optuna는 이 값을 최대화하는 방향으로 다음 파라미터를 탐색합니다.
    return best_accuracy

def main():
    # >> argparse를 사용하여 커맨드 라인 인자를 설정한다.
    parser = argparse.ArgumentParser(description="Train GCN-Transformer model.")
    parser.add_argument(
        '--scheduler', 
        type=str, 
        default='cosine_decay', 
        choices=['cosine_decay', 'cosine_restarts'],
        help="Scheduler to use: 'cosine_decay' or 'cosine_restarts'"
    )
    parser.add_argument(
        '-p', '--protocol', type=str, default='xsub', choices=['xsub', 'xview'],
        help="Training protocol: 'xsub' (Cross-Subject) or 'xview' (Cross-View). Default: 'xsub'"
    )
    # --- [OPTUNA] --- 연구 이름을 받아 이어서 할 수 있도록 추가
    parser.add_argument('--study-name', type=str, default="slowfast_gcn_tuning",
                        help="Name for the Optuna study.")
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                        help="Total number of trials to run.")
    
    args = parser.parse_args()

    print(f"--- Starting Optuna Study ---")
    print(f"Study Name: {args.study_name}")
    print(f"Protocol: {args.protocol}")
    print(f"Total Trials: {args.n_trials}")
    print(f"Epochs per Trial: {MAX_EPOCHS_PER_TRIAL}")
    print(f"-----------------------------")

    # --- [OPTUNA] 1. Study 생성 ---
    # Pruner: 중간 결과가 나쁜 trial을 조기 중단 (가지치기)
    # MedianPruner: 다른 trial들의 '중간값'보다 성능이 나쁘면 중단
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, # 처음 5개 trial은 가지치기 안 함
        n_warmup_steps=PRUNING_WARMUP_EPOCHS, # 10 에폭 전까지는 가지치기 안 함
        interval_steps=1 # 1 에폭마다 가지치기 여부 확인
    )
    
    # Storage: 모든 실험 결과를 SQLite 데이터베이스에 저장
    # (스크립트가 중단되어도 'optuna_study.db' 파일에 기록이 남아 이어서 할 수 있음)
    storage_name = f"sqlite:///{args.study_name}.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        direction='maximize', # 우리는 'Val Acc'를 '최대화'하는 것이 목표
        pruner=pruner,
        load_if_exists=True # 이전에 중단된 study가 있으면 이어서 실행
    )

    # --- [OPTUNA] 2. Study 실행 ---
    # study.optimize는 `run_trial` 함수를 `n_trials` 횟수만큼 호출합니다.
    # lambda 함수를 사용해 `run_trial`에 `args` 인자를 전달합니다.
    try:
        study.optimize(
            lambda trial: run_trial(trial, args), 
            n_trials=args.n_trials
        )
    except KeyboardInterrupt:
        print("\nUser interrupted the Optuna study. Saving results...")

    # --- [OPTUNA] 3. 최종 결과 리포트 ---
    print("\n--- Optuna Study Finished ---")
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"Study statistics: ")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")

    print("\n--- Best Trial ---")
    best_trial = study.best_trial
    print(f"  Value (Best Val Acc): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        # 값의 형태에 따라 소수점 포맷팅
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
            
    print(f"\nBest model saved in: {config.SAVE_DIR}/trial_{best_trial.number}/best_model.pth.tar")
    print(f"To resume study, run the same command again.")
    print(f"To analyze results, run: optuna-dashboard {storage_name}")


if __name__ == '__main__':
    # --- [OPTUNA] --- main() 함수가 Optuna 로직을 실행하도록 변경됨
    main()
