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

# >> ASK: 새 Trail을 시작하고 싶을 때 터미널에서 실행하는 명령어
# python manager.py --study-name my_study2 ask

# >> Train: 위 명령어가 출력해준 python train.py ... 명령어를 복사하여 터미널에 붙여넣고 실행하기
# 예시: python train.py --protocol xsub --scheduler ... --trial-number 5 --lr 0.000123 ...

# >> 중단/재시작: 훈련 기간 중 Ctrl+c를 눌러서 중단해도, 위에서 입력한 python train.py ... 명령어를 다\
시 실행하면 체크포인트를 불러와서 이어서 학습한다.

# >> 종료: 훈련이 특정 epoch까지 모두 완료되면 터미널에서 최종 안내가 나온다.
# To report this result to Optuna, run the following command:
# python manager.py tell --trial-number 5 --value 0.9123

# >> Tell: 위 문구를 복사하여 터미널에서 실행
# python manager.py --study-name my_study1 --tell --trial-number 5 --value 0.9123

# >> ASK: 다음 트라이얼을 위해 ASK 부터 반복한다.

# >> 실시간으로 현황 보기
# optuna-dashboard sqlite:///my_study1.db



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
# import optuna
import traceback
import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import SlowFast_GCNTransformer
from utils import calculate_accuracy, save_checkpoint, load_checkpoint


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
MAX_EPOCHS_PER_TRIAL = config.EPOCHS 





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

            # 1. Action 손실 계산 (이제 'mean'으로 바로 계산됨)
            loss_action = criterion_action(outputs_action, action_labels)
            
            # 2. Subject 손실 계산 (기존과 동일)
            loss_subject = criterion_subject(outputs_subject, subject_labels)

            
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
                outputs_action, outputs_subject = model(data_fast, data_slow)
                loss_action = criterion_action(outputs_action, action_labels)
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



# Optuna의 'trial' 객체를 인자에서 제거합니다.
# 대신 커맨드 라인 인자인 'args'를 받습니다.
def run_trial(args):
    try: # KeyboardInterrupt(Ctrl+C)를 잡기 위해 try 블록 시작
        """단일 트라이얼을 실행하는 함수"""
        
        # >> 재현성을 위해 시드를 고정한다.
        set_seed(config.SEED)


        # 1. Optuna의 trial.suggest_... 대신 args에서 하이퍼파라미터를 받아
        #    config.py의 값을 덮어씁니다.
        config.LEARNING_RATE = args.lr
        config.DROPOUT = args.dropout
        config.ADVERSARIAL_ALPHA = args.alpha
        config.PROB = args.prob
        config.ADAMW_WEIGHT_DECAY = args.weight_decay
        config.LABEL_SMOOTHING = args.smoothing
        

        # 트라이얼 번호를 args에서 가져옵니다.
        trial_number = args.trial_number
        
        print(f"\n--- [Running Trial {trial_number}] ---")
        print(f"Params: LR={config.LEARNING_RATE:.6f}, Dropout={config.DROPOUT:.3f}, Alpha={config.ADVERSARIAL_ALPHA:.3f}")
        print(f"        Prob={config.PROB:.3f}, WeightDecay={config.ADAMW_WEIGHT_DECAY:.4f}, Smoothing={config.LABEL_SMOOTHING:.3f}")

        # --- 2. 학습 준비 ---
        device = config.DEVICE

        # (데이터 로더는 덮어쓴 config.PROB 값 등을 참조하여 생성됩니다)
        train_dataset = NTURGBDDataset(
            data_path = config.DATASET_PATH, split = 'train',
            max_frames = config.MAX_FRAMES, protocol = args.protocol
        )
        train_loader = DataLoader(
            train_dataset, batch_size = config.BATCH_SIZE, shuffle = True,
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
            label_smoothing=config.LABEL_SMOOTHING, reduction='mean'
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
            total_epochs=MAX_EPOCHS_PER_TRIAL, # config.EPOCHS (예: 100)
            warmup_epochs=config.WARMUP_EPOCHS
        )

        best_accuracy = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


        # 트라이얼 번호(args.trial_number)를 기반으로 고유 폴더 생성
        trial_save_dir = os.path.join(config.SAVE_DIR, f"trial_{trial_number}")
        os.makedirs(trial_save_dir, exist_ok=True)

        # --- 체크포인트 경로를 '최고 성능'과 '재시작용'으로 분리 ---
        best_model_path = os.path.join(trial_save_dir, "best_model.pth.tar")
        resume_checkpoint_path = os.path.join(trial_save_dir, "resume_checkpoint.pth.tar")


        # 1. 변수 기본값 초기화 (새로 시작할 경우)
        start_epoch = 0
        best_accuracy = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        patience_counter = 0
        lr_drop_count = 0

        # 2. "재시작용" 체크포인트 파일이 "존재"하는지 확인
        if os.path.exists(resume_checkpoint_path):
            print(f"\n--- [Trial {trial_number}] Resuming from checkpoint: {resume_checkpoint_path} ---")
            try:
                # utils.py의 load_checkpoint 함수 사용
                checkpoint = load_checkpoint(
                    resume_checkpoint_path, model, optimizer, scheduler, device
                ) #
                
                # 4. 저장된 모든 상태 복원
                start_epoch = checkpoint['epoch'] # <--- 이 값이 14라면, range(14, 100)이 됨
                best_accuracy = checkpoint['best_acc']
                history = checkpoint.get('history', history) 
                patience_counter = checkpoint.get('patience_counter', patience_counter)
                lr_drop_count = checkpoint.get('lr_drop_count', lr_drop_count)
                
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                
                print(f"Resumed from Epoch {start_epoch}. Best Acc so far: {best_accuracy:.4f}")

            except Exception as e:
                print(f"Error loading resume checkpoint for Trial {trial_number}: {e}")
                print("Starting from Epoch 0.")
                start_epoch = 0 # 로드 실패 시 0부터 다시 시작
        else:
            print(f"\n--- [Trial {trial_number}] No resume checkpoint found. Starting from Epoch 0. ---")
        
        

        # config.EPOCHS (예: 100) 만큼 학습합니다.

        for epoch in range(start_epoch, MAX_EPOCHS_PER_TRIAL):
            epoch_start_time = time.time()
            print(f"\n--- [Trial {trial_number}] Epoch {epoch+1}/{MAX_EPOCHS_PER_TRIAL} ---")
        
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
                print(f"New best accuracy for Trial {trial_number}: {val_acc_action:.4f}! Saving model...")
                

                # 체크포인트에 Optuna의 'trial.params' 대신, 
                # 'args'로부터 받은 파라미터 딕셔너리를 저장합니다.
                optuna_params_dict = {
                    "LEARNING_RATE": args.lr,
                    "DROPOUT": args.dropout,
                    "ADVERSARIAL_ALPHA": args.alpha,
                    "PROB": args.prob,
                    "ADAMW_WEIGHT_DECAY": args.weight_decay,
                    "LABEL_SMOOTHING": args.smoothing
                }

                
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
                    'optuna_params': optuna_params_dict # --- [Ask-Tell 수정] ---
                }, directory=trial_save_dir, filename="best_model.pth.tar")

            else: 
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}. LR Drops: {lr_drop_count}.")
                
                if patience_counter >= config.PATIENCE: 
                    print(f"Early stopping triggered for Trial {trial_number} (Patience: {config.PATIENCE}).")
                    break # 이 Trial의 학습 루프 조기 종료
                    

            # 재시작 체크포인트에도 수정된 파라미터 딕셔너리를 저장합니다.
            optuna_params_dict = {
                "LEARNING_RATE": args.lr,
                "DROPOUT": args.dropout,
                "ADVERSARIAL_ALPHA": args.alpha,
                "PROB": args.prob,
                "ADAMW_WEIGHT_DECAY": args.weight_decay,
                "LABEL_SMOOTHING": args.smoothing
            }

            save_checkpoint({
                'epoch': epoch + 1, # <--- 다음 에폭 번호(예: 14+1=15)를 저장
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_accuracy, # 현재까지의 최고 성능
                'scaler': scaler.state_dict(),
                'history': history,
                'patience_counter': patience_counter, # 현재 Patience
                'lr_drop_count': lr_drop_count,
                'optuna_params': optuna_params_dict # --- [Ask-Tell 수정] ---
            }, directory=trial_save_dir, filename="resume_checkpoint.pth.tar")
                


        
        # 그래프를 trial마다 저장합니다.
        plot_history(history, os.path.join(trial_save_dir, "training_history.png"))
        
        # 이 trial의 최종 '최고 정확도'를 반환합니다.
        return best_accuracy
    
    except KeyboardInterrupt: # <<< [Ask-Tell 수정] ---
        print("\n\n-------------------------------------------------")
        print(f"     [Trial {args.trial_number} 중단됨]")
        print("       체크포인트가 저장되었습니다.")
        print("    동일한 명령어를 다시 실행하면 이어서 학습합니다.")
        print("-------------------------------------------------")
        
        # Optuna 스터디의 일부가 아니므로, 스크립트를 정상 종료(0)시킵니다.
        sys.exit(0)

        



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
    

    # Optuna 스터디 관련 인자('--study-name', '--n-trials')를 제거하고,
    # 하이퍼파라미터 인자를 추가합니다.
    parser.add_argument('--study-name', type=str, required=True,
                        help="Optuna study name (provided by manager.py --ask)")
    parser.add_argument('--trial-number', type=int, required=True,
                        help="Optuna trial number (provided by manager.py --ask)")
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help="Learning Rate")
    parser.add_argument('--dropout', type=float, default=config.DROPOUT,
                        help="Dropout rate")
    parser.add_argument('--alpha', type=float, default=config.ADVERSARIAL_ALPHA,
                        help="Adversarial alpha (GRL strength)")
    parser.add_argument('--prob', type=float, default=config.PROB,
                        help="Data augmentation 'skip' probability")
    parser.add_argument('--weight-decay', type=float, default=config.ADAMW_WEIGHT_DECAY,
                        help="AdamW weight decay")
    parser.add_argument('--smoothing', type=float, default=config.LABEL_SMOOTHING,
                        help="Label smoothing value")
    
    args = parser.parse_args()

    best_acc = run_trial(args) 
    

    # 훈련이 완료되면, 사용자에게 결과를 Optuna에 보고하라고 안내합니다.
    print(f"\n--- [Trial {args.trial_number}] Finished ---")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"\n[Action Required]")
    print(f"To report this result to Optuna, run the following command:")
    print(f"python manager.py --study-name {args.study_name} tell --trial-number {args.trial_number} --value {best_acc:.4f}")
    

if __name__ == '__main__':
    # main() 함수는 이제 Optuna 대신 단일 트라이얼을 실행합니다.
    main()
