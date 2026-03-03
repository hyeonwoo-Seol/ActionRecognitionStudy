# train.py

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
import time
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
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    
    if scheduler_name == 'cosine_decay':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=config.ETA_MIN)
    elif scheduler_name == 'cosine_restarts':
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_MULT, eta_min=config.ETA_MIN)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])


def get_current_alpha(epoch, max_alpha=0.2):
    warmup_epochs = config.WARMUP_EPOCHS
    total_epochs = MAX_EPOCHS_PER_TRIAL
    if epoch < warmup_epochs: return 0.0
    if total_epochs <= warmup_epochs: return max_alpha 
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    return max_alpha * progress


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
    plt.close()


def train_one_epoch(model, loader, criterion_action, criterion_aux, optimizer, device, scaler, epoch):
    model.train()
    
    # Optimizer에서 현재 LR 가져오기
    current_lr = optimizer.param_groups[0]['lr']
    current_alpha = get_current_alpha(epoch, max_alpha=config.ADVERSARIAL_ALPHA)
    
    if hasattr(model, 'grad_reversal'):
        model.grad_reversal.alpha = current_alpha
    
    running_loss = 0.0
    correct_action = 0
    correct_aux = 0
    total_samples = 0

    # 진행률 표시줄에 현재 LR과 Alpha 표시
    desc_str = f"[Train Ep {epoch+1}/{MAX_EPOCHS_PER_TRIAL}] LR={current_lr:.6f} | α={current_alpha:.3f}"
    train_bar = tqdm(loader, desc=desc_str, colour="green", leave=False)
    
    for data_fast, data_slow, action_labels, aux_labels in train_bar:
        data_fast = data_fast.to(device)
        data_slow = data_slow.to(device)
        action_labels = action_labels.to(device)
        aux_labels = aux_labels.to(device)
        
        optimizer.zero_grad()

        with autocast(device_type=device):
            outputs_action, outputs_aux = model(data_fast, data_slow)

            loss_action = criterion_action(outputs_action, action_labels)
            loss_aux = criterion_aux(outputs_aux, aux_labels)
            
            loss = loss_action + loss_aux

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * data_fast.size(0)
        total_samples += action_labels.size(0)
        
        _, predicted_action = torch.max(outputs_action.data, 1)
        correct_action += (predicted_action == action_labels).sum().item()
        
        _, predicted_aux = torch.max(outputs_aux.data, 1)
        correct_aux += (predicted_aux == aux_labels).sum().item()
        
        train_bar.set_postfix(
            loss=f"{running_loss/total_samples:.4f}",
            acc_ACT=f"{correct_action/total_samples:.4f}",
            acc_AUX=f"{correct_aux/total_samples:.4f}"
        )
        
    return (running_loss / total_samples, correct_action / total_samples, correct_aux / total_samples)


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
                outputs_action, _ = model(data_fast, data_slow)
                loss_action = criterion_action(outputs_action, action_labels)
                loss = loss_action 
                
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
        config.LEARNING_RATE = args.lr
        config.DROPOUT = args.dropout
        config.ADVERSARIAL_ALPHA = args.alpha
        config.PROB = args.prob
        config.ADAMW_WEIGHT_DECAY = args.weight_decay
        config.LABEL_SMOOTHING = args.smoothing
        
        print(f"\n--- [Running Trial {args.trial_number}] Protocol: {args.protocol} ---")

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

        # Protocol에 따른 Aux Class 개수 설정
        if args.protocol == 'xsub':
            num_aux_classes = config.NUM_SUBJECTS
            print(f" >> GRL Target: Subject Classification ({num_aux_classes} classes)")
        elif args.protocol == 'xview':
            num_aux_classes = config.NUM_CAMERAS
            print(f" >> GRL Target: Camera Viewpoint Classification ({num_aux_classes} classes)")

        model = SlowFast_Transformer(
            num_joints=config.NUM_JOINTS,
            num_coords=config.NUM_COORDS,
            num_classes=config.NUM_CLASSES,
            num_aux_classes=num_aux_classes,
            alpha=0.0 
        ).to(device)

        criterion_action = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        criterion_aux = nn.CrossEntropyLoss() # GRL Loss
        
        optimizer = optim.AdamW(
            model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.ADAMW_WEIGHT_DECAY
        )
        scaler = GradScaler()
        scheduler = get_scheduler(
            optimizer, args.scheduler, MAX_EPOCHS_PER_TRIAL, config.WARMUP_EPOCHS
        )

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        trial_save_dir = os.path.join(config.SAVE_DIR, f"trial_{args.trial_number}")
        os.makedirs(trial_save_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(trial_save_dir, "resume_checkpoint.pth.tar")

        start_epoch = 0
        best_accuracy = 0.0
        patience_counter = 0
        
        if os.path.exists(resume_checkpoint_path):
            checkpoint = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, device)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_acc']
            history = checkpoint.get('history', history)
            if 'scaler' in checkpoint: scaler.load_state_dict(checkpoint['scaler'])

        # --- Training Loop ---
        for epoch in range(start_epoch, MAX_EPOCHS_PER_TRIAL):
            epoch_start_time = time.time()
            
            current_lr_for_log = optimizer.param_groups[0]['lr']
            current_alpha_for_log = get_current_alpha(epoch, max_alpha=config.ADVERSARIAL_ALPHA)
            
            train_loss, train_acc_action, train_acc_aux = train_one_epoch(
                model, train_loader, criterion_action, criterion_aux, optimizer, device, scaler, epoch
            )
            
            val_loss, val_acc_action = validate_one_epoch(
                model, val_loader, criterion_action, device
            )

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc_action)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc_action)
            
            # Epoch 종료 후 스케줄러 업데이트
            scheduler.step()
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch [{epoch+1}/{MAX_EPOCHS_PER_TRIAL}] | Time: {epoch_time:.1f}s | LR: {current_lr_for_log:.6f} | Alpha: {current_alpha_for_log:.4f}")
            print(f" >> [ACT] Train: {train_acc_action:.4f} | Val: {val_acc_action:.4f}")
            print(f" >> [AUX] Train: {train_acc_aux:.4f} (GRL Target)")

            if val_acc_action > best_accuracy:
                best_accuracy = val_acc_action
                patience_counter = 0 
                save_checkpoint({
                    'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'best_acc': best_accuracy, 'history': history,
                }, directory=trial_save_dir, filename="best_model.pth.tar")
            else: 
                patience_counter += 1
                if patience_counter >= config.PATIENCE: 
                    print("Early stopping.")
                    break 

            save_checkpoint({
                'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'best_acc': best_accuracy, 'scaler': scaler.state_dict(), 'history': history
            }, directory=trial_save_dir, filename="resume_checkpoint.pth.tar")
        
        plot_history(history, os.path.join(trial_save_dir, "training_history.png"))
        return best_accuracy
    
    except KeyboardInterrupt:
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler', type=str, default='cosine_decay', choices=['cosine_decay', 'cosine_restarts'])
    parser.add_argument('-p', '--protocol', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--study-name', type=str, required=True)
    parser.add_argument('--trial-number', type=int, required=True)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--dropout', type=float, default=config.DROPOUT)
    parser.add_argument('--alpha', type=float, default=config.ADVERSARIAL_ALPHA)
    parser.add_argument('--prob', type=float, default=config.PROB)
    parser.add_argument('--weight-decay', type=float, default=config.ADAMW_WEIGHT_DECAY)
    parser.add_argument('--smoothing', type=float, default=config.LABEL_SMOOTHING)
    
    args = parser.parse_args()
    best_acc = run_trial(args) 
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    # print(f"python manager.py --study-name {args.study_name} tell --trial-number {args.trial_number} --value {best_acc:.4f}")

if __name__ == '__main__':
    main()
