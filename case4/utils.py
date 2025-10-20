# utils.py

import torch
import os

def calculate_accuracy(outputs, labels):
    # 모델의 출력(logits)과 실제 레이블을 기반으로 정확도를 계산 
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def save_checkpoint(state, directory, filename="checkpoint.pth.tar"):
    # 체크포인트 저장 
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    # 저장된 체크포인트 복원
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    # 장치에 맞게 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print(f"Model checkpoint loaded from '{checkpoint_path}'")
    return checkpoint
