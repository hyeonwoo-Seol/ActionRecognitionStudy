# train.py

# ## --------------------------------------------------------------------------
# GRL 제거 및 Standalone(단독) 실행 지원 버전
# python train.py 만 입력하면 config.py 설정대로 학습합니다.
# Optuna 사용 시 기존처럼 인자를 주면 됩니다.
# ## --------------------------------------------------------------------------

import torch
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
import traceback
import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import SlowFast_Transformer
from utils import calculate_accuracy, save_checkpoint, load_checkpoint

MAX_EPOCHS_PER_TRIAL = config.EPOCHS 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_scheduler(optimizer, scheduler_name, total_epochs, warmup_epochs):
    print(f"Using '{scheduler_name}' scheduler.")
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    if scheduler_name == 'cosine_decay':
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=config.ETA_MIN
        )
    elif scheduler_name == 'cosine_restarts':
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_MULT,
            eta_min=config.ETA_MIN
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_history(history, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    epochs = range(1, len(history['train_acc']) + 1)
    ax1.plot(epochs, history['train_acc'], 'g-', label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'], 'b-', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_loss'], 'r--', label='Train Loss')
    ax2.plot(epochs, history['val_loss'], 'm--', label='Validation Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.suptitle('Training and Validation History', fontsize=16)
    fig.legend(loc='upper left', bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
    plt.savefig(save_path)
    print(f"\nTraining history graph saved to '{save_path}'")
    plt.close()

def train_one_epoch(model, loader, criterion_action, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct_action = 0
    total_samples = 0

    train_bar = tqdm(loader, desc="[Train]", colour="green", leave=False)
    for data_fast, data_slow, action_labels, _ in train_bar: 
        data_fast = data_fast.to(device)
        data_slow = data_slow.to(device)
        action_labels = action_labels.to(device)
        
        optimizer.zero_grad()

        with autocast(device_type=device):
            outputs_action = model(data_fast, data_slow)
            loss_action = criterion_action(outputs_action, action_labels)
            loss = loss_action 

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * data_fast.size(0)
        total_samples += action_labels.size(0)
        
        _, predicted_action = torch.max(outputs_action.data, 1)
        correct_action += (predicted_action == action_labels).sum().item()
        
        train_bar.set_postfix(
            loss=f"{running_loss/total_samples:.4f}",
            acc_ACT=f"{correct_action/total_samples:.4f}"
        )
        
    return (running_loss / total_samples, correct_action / total_samples)


def validate_one_epoch(model, loader, criterion_action, device):
    model.eval()
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    val_bar = tqdm(loader, desc="[Val]", colour="cyan", leave=False)

    with torch.no_grad():
        for data_fast, data_slow, action_labels, _ in val_bar:
            data_fast = data_fast.to(device)
            data_slow = data_slow.to(device)
            action_labels = action_labels.to(device)

            with autocast(device_type=device):
                outputs_action = model(data_fast, data_slow)
                loss = criterion_action(outputs_action, action_labels)
                
            running_loss += loss.item() * data_fast.size(0)
            total_samples += action_labels.size(0)
            
            _, predicted_action = torch.max(outputs_action.data, 1)
            correct_action += (predicted_action == action_labels).sum().item()
            
            val_bar.set_postfix(
                loss=f"{running_loss/total_samples:.4f}",
                acc_ACT=f"{correct_action/total_samples:.4f}"
            )
    
    return (running_loss / total_samples, correct_action / total_samples)


def run_trial(args):
    try: 
        set_seed(config.SEED)

        # args에 값이 있으면(사용자가 지정했거나 기본값) config를 덮어씁니다.
        # python train.py만 실행하면 args.lr에는 config.LEARNING_RATE가 기본값으로 들어있으므로
        # 사실상 config 값 그대로 유지됩니다.
        config.LEARNING_RATE = args.lr
        config.DROPOUT = args.dropout
        config.PROB = args.prob
        config.ADAMW_WEIGHT_DECAY = args.weight_decay
        config.LABEL_SMOOTHING = args.smoothing
        
        # [수정] Standalone 모드 처리
        if args.trial_number is None:
            trial_identifier = "standalone"
            print("\n--- [Running Standalone Training] ---")
        else:
            trial_identifier = args.trial_number
            print(f"\n--- [Running Trial {trial_identifier} (No GRL)] ---")
            
        print(f"Params: LR={config.LEARNING_RATE:.6f}, Dropout={config.DROPOUT:.3f}")

        device = config.DEVICE

        train_dataset = NTURGBDDataset(
            data_path = config.DATASET_PATH, split = 'train',
            max_frames = config.MAX_FRAMES, protocol = args.protocol
        )
        g = torch.Generator()
        g.manual_seed(config.SEED)
        
        train_loader = DataLoader(
            train_dataset, batch_size = config.BATCH_SIZE, shuffle = True,
            num_workers = config.NUM_WORKERS, pin_memory = config.PIN_MEMORY,
            worker_init_fn=seed_worker, generator=g
        )
        val_dataset = NTURGBDDataset(
            data_path = config.DATASET_PATH, split = 'val',
            max_frames = config.MAX_FRAMES, protocol = args.protocol
        )
        val_loader = DataLoader(
            val_dataset, batch_size = config.BATCH_SIZE, shuffle = False,
            num_workers = config.NUM_WORKERS, pin_memory = config.PIN_MEMORY,
            worker_init_fn=seed_worker, generator=g
        )

        model = SlowFast_Transformer(
            num_joints=config.NUM_JOINTS,
            num_coords=config.NUM_COORDS,
            num_classes=config.NUM_CLASSES,
            fast_dims=config.FAST_DIMS,
            slow_dims=config.SLOW_DIMS,
        ).to(device)

        criterion_action = nn.CrossEntropyLoss(
            label_smoothing=config.LABEL_SMOOTHING, reduction='mean'
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr = config.LEARNING_RATE,
            weight_decay = config.ADAMW_WEIGHT_DECAY
        )
        scaler = GradScaler()
        scheduler = get_scheduler(
            optimizer,
            scheduler_name=args.scheduler,
            total_epochs=MAX_EPOCHS_PER_TRIAL,
            warmup_epochs=config.WARMUP_EPOCHS
        )

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # [수정] 저장 폴더 이름 설정
        trial_save_dir = os.path.join(config.SAVE_DIR, f"trial_{trial_identifier}")
        os.makedirs(trial_save_dir, exist_ok=True)

        best_model_path = os.path.join(trial_save_dir, "best_model.pth.tar")
        resume_checkpoint_path = os.path.join(trial_save_dir, "resume_checkpoint.pth.tar")

        start_epoch = 0
        best_accuracy = 0.0
        patience_counter = 0
        lr_drop_count = 0

        if os.path.exists(resume_checkpoint_path):
            print(f"\n--- Resuming from checkpoint: {resume_checkpoint_path} ---")
            try:
                checkpoint = load_checkpoint(
                    resume_checkpoint_path, model, optimizer, scheduler, device
                ) 
                start_epoch = checkpoint['epoch'] 
                best_accuracy = checkpoint['best_acc']
                history = checkpoint.get('history', history) 
                patience_counter = checkpoint.get('patience_counter', patience_counter)
                lr_drop_count = checkpoint.get('lr_drop_count', lr_drop_count)
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                print(f"Resumed from Epoch {start_epoch}. Best Acc so far: {best_accuracy:.4f}")
            except Exception as e:
                print(f"Error loading resume checkpoint: {e}. Starting from Epoch 0.")
                start_epoch = 0
        else:
            print(f"\n--- Starting from Epoch 0. ---")
        
        for epoch in range(start_epoch, MAX_EPOCHS_PER_TRIAL):
            epoch_start_time = time.time()
            # print(f"\n--- Epoch {epoch+1}/{MAX_EPOCHS_PER_TRIAL} ---")
        
            train_loss, train_acc_action = train_one_epoch(
                model, train_loader, criterion_action, optimizer, device, scaler
            )
            
            val_loss, val_acc_action = validate_one_epoch(
                model, val_loader, criterion_action, device
            )

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc_action)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc_action)
                
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
                
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{MAX_EPOCHS_PER_TRIAL} | Train Acc: {train_acc_action:.4f} | Val Acc: {val_acc_action:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

            if val_acc_action > best_accuracy:
                best_accuracy = val_acc_action
                patience_counter = 0 
                print(f"New best accuracy: {val_acc_action:.4f}! Saving model...")
                
                optuna_params_dict = {
                    "LEARNING_RATE": args.lr,
                    "DROPOUT": args.dropout,
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
                    'optuna_params': optuna_params_dict 
                }, directory=trial_save_dir, filename="best_model.pth.tar")

            else: 
                patience_counter += 1
                # print(f"No improvement. Patience: {patience_counter}.")
                
                if patience_counter >= config.PATIENCE: 
                    print(f"Early stopping triggered.")
                    break 

            optuna_params_dict = {
                "LEARNING_RATE": args.lr,
                "DROPOUT": args.dropout,
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
                'optuna_params': optuna_params_dict
            }, directory=trial_save_dir, filename="resume_checkpoint.pth.tar")
                
        plot_history(history, os.path.join(trial_save_dir, "training_history.png"))
        
        return best_accuracy
    
    except KeyboardInterrupt:
        print("\n\n-------------------------------------------------")
        print(f"     [Trial {trial_identifier} 중단됨]")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Train Transformer model (No GRL).")
    parser.add_argument(
        '--scheduler', type=str, default='cosine_decay', 
        choices=['cosine_decay', 'cosine_restarts']
    )
    parser.add_argument(
        '-p', '--protocol', type=str, default='xsub', choices=['xsub', 'xview']
    )

    # [수정] Optuna 관련 인자를 Optional(선택)로 변경하고 default=None 설정
    parser.add_argument('--study-name', type=str, required=False, default=None,
                        help="Optuna study name (Optional for standalone training)")
    parser.add_argument('--trial-number', type=int, required=False, default=None,
                        help="Optuna trial number (Optional for standalone training)")
    
    # 하이퍼파라미터 인자들의 기본값을 config.py의 값으로 설정
    # 사용자가 인자를 입력하지 않으면 config 값이 사용됨
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--dropout', type=float, default=config.DROPOUT)
    parser.add_argument('--prob', type=float, default=config.PROB)
    parser.add_argument('--weight-decay', type=float, default=config.ADAMW_WEIGHT_DECAY)
    parser.add_argument('--smoothing', type=float, default=config.LABEL_SMOOTHING)
    
    parser.add_argument('--alpha', type=float, default=0.0, help="Deprecated (GRL removed)")

    args = parser.parse_args()

    best_acc = run_trial(args) 

    # [수정] Standalone 모드일 때는 Optuna 안내 메시지를 출력하지 않음
    if args.trial_number is not None:
        print(f"\n--- [Trial {args.trial_number}] Finished ---")
        print(f"Best Validation Accuracy: {best_acc:.4f}")
        if args.study_name:
            print(f"python manager.py --study-name {args.study_name} tell --trial-number {args.trial_number} --value {best_acc:.4f}")
    else:
        print(f"\n--- [Standalone Training] Finished ---")
        print(f"Best Validation Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()
