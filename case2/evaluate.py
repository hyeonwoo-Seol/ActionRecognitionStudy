# evaluate.py

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

# 직접 작성한 파일들 임포트
import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import GCNMambaModel
from utils import load_checkpoint

def evaluate_model(checkpoint_path):
    """
    저장된 모델을 불러와 검증 데이터셋으로 성능을 평가합니다.
    """
    # --- 설정값 불러오기 ---
    print(f"Using device: {config.DEVICE}")
    
    # --- 데이터 로딩 ---
    # 평가 시에는 데이터 순서가 중요하지 않으므로 shuffle=False로 설정합니다.
    val_dataset = NTURGBDDataset(data_path=config.DATASET_PATH, split='val', max_frames=config.MAX_FRAMES)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 모델 초기화 및 체크포인트 로드 ---
    model = GCNMambaModel(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    load_checkpoint(checkpoint_path, model, device=config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()

    # --- 평가 루프 ---
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    eval_bar = tqdm(val_loader, desc="[Evaluate]", colour="yellow")
    with torch.no_grad():
        for motion_features, labels, first_frame_coords in eval_bar:
            motion_features = motion_features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            first_frame_coords = first_frame_coords.to(config.DEVICE)
            
            # 💡 <<-- 변경점: 모델에 2개의 인자 전달 -->>
            outputs = model(motion_features, first_frame_coords)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * motion_features.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            avg_loss = total_loss / total_samples
            avg_acc = correct_predictions / total_samples
            eval_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")


    # --- 최종 결과 출력 ---
    final_loss = total_loss / len(val_dataset)
    final_accuracy = correct_predictions / len(val_dataset)
    
    print("\n--- Evaluation Finished ---")
    print(f"Average Loss: {final_loss:.4f}")
    print(f"Accuracy: {final_accuracy * 100:.2f}%")

if __name__ == '__main__':
    # 커맨드 라인에서 체크포인트 경로를 받을 수 있도록 설정
    parser = argparse.ArgumentParser(description="Evaluate the ActionMamba model.")
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint file (e.g., 'checkpoints/best_model.pth.tar')")
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint)
