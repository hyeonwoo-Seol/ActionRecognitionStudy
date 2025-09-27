# pretrain_mae.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from torch.amp import GradScaler, autocast


import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import GCNTransformerModel
from utils import save_checkpoint
from utils import save_checkpoint, load_checkpoint

def main():
    config.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    pretrain_dataset = NTURGBDDataset(
        data_path=config.DATASET_PATH, # .pt 파일들이 있는 디렉토리
        split='pretrain',
        max_frames=config.MAX_FRAMES
        # indices=all_indices # 더 이상 필요 없음
    )
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )


    # --- 2. 모델 초기화 (mode='pretrain') ---
    model = GCNTransformerModel(
        block_type=config.BLOCK_TYPE,
        layer_dims=config.LAYER_DIMS,
        use_gcn=config.USE_GCN,
        mode='pretrain' # <<-- 사전학습 모드로 설정
    ).to(device)

    # --- 3. 손실 함수, 옵티마이저, 스케줄러 설정 ---
    criterion = nn.MSELoss() # 복원 문제이므로 MSE 사용
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.ADAMW_WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.ETA_MIN)

    scaler = GradScaler()

    # <<-- 1. 체크포인트 경로 설정 및 상태 변수 초기화 -->>
    best_checkpoint_path = os.path.join(config.SAVE_DIR, "mae_pretrained_best.pth.tar")
    latest_checkpoint_path = os.path.join(config.SAVE_DIR, "mae_pretrained_latest.pth.tar")
    
    start_epoch = 0
    best_loss = float('inf')

    # <<-- 2. 'latest' 체크포인트가 존재하면 상태 복원 -->>
    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming pre-training from checkpoint '{latest_checkpoint_path}'...")
        # load_checkpoint 함수는 모델과 옵티마이저의 상태를 내부적으로 로드합니다.
        checkpoint = load_checkpoint(latest_checkpoint_path, model, optimizer, device)
        
        # 저장된 epoch, best_loss, scheduler, scaler 상태를 복원합니다.
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Scheduler state loaded.")
            
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            print("GradScaler state loaded.")
            
        print(f"Resumed from epoch {start_epoch}. Current best loss is {best_loss:.6f}")

    # <<-- 3. 학습 루프 시작점을 start_epoch으로 수정 -->>
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(pretrain_loader, desc=f"[Pretrain Epoch {epoch+1}]", colour="yellow")
        for masked_data, original_data, mask in train_bar:
            masked_data = masked_data.to(device)
            original_data = original_data.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device):
                reconstructed_features = model(masked_data, mask=mask)
                original_masked_parts = original_data.permute(0, 2, 3, 1)[mask]
                loss = criterion(reconstructed_features, original_masked_parts)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{running_loss / len(train_bar):.6f}")

        epoch_loss = running_loss / len(pretrain_loader)
        scheduler.step()

        # <<-- 4. 체크포인트 저장 전략 수정 -->>
        # 최고 성능 모델 저장 (기존 로직 유지)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"New best pre-training loss: {best_loss:.6f}! Saving best model...")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_loss': best_loss
            }, directory=config.SAVE_DIR, filename="mae_pretrained_best.pth.tar")

        # 매 epoch 마다 최신 상태 저장 (재개를 위한 로직)
        print("Saving latest checkpoint...")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_loss': best_loss
        }, directory=config.SAVE_DIR, filename="mae_pretrained_latest.pth.tar")


    print("\nPre-training finished.")
    

if __name__ == '__main__':
    from train import set_seed
    config.set_seed = set_seed
    main()
