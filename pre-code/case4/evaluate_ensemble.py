# evaluate_ensemble.py (새 파일)

# python evaluate_ensemble.py --checkpoint_dir checkpoints/

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import glob  # 파일 경로를 쉽게 찾기 위해 glob 라이브러리 임포트

# 직접 작성한 파일들 임포트
import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import GCNTransformerModel # 모델 이름은 GCNTransformerModel로 수정
from utils import load_checkpoint

def evaluate_ensemble(checkpoint_dir):
    """
    지정된 디렉토리의 모든 스냅샷 모델을 불러와 앙상블 평가를 수행합니다.
    """
    # --- 설정값 불러오기 ---
    print(f"Using device: {config.DEVICE}")
    
    # --- 데이터 로딩 ---
    val_dataset = NTURGBDDataset(data_path=config.DATASET_PATH, split='val', max_frames=config.MAX_FRAMES)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 모델 초기화 ---
    # 모델은 한 번만 생성하고, 가중치만 바꿔가며 로드합니다.
    model = GCNTransformerModel(
        block_type=config.BLOCK_TYPE,
        layer_dims=config.LAYER_DIMS,
        use_gcn=config.USE_GCN
    ).to(config.DEVICE)
    model.eval()
    
    # --- 평가할 스냅샷 파일 목록 찾기 ---
    # 'snapshot_epoch_*.pth.tar' 패턴에 맞는 모든 파일을 찾습니다.
    snapshot_paths = glob.glob(os.path.join(checkpoint_dir, 'snapshot_epoch_*.pth.tar'))
    
    if not snapshot_paths:
        print(f"Error: No snapshot files found in '{checkpoint_dir}'")
        return
        
    print(f"Found {len(snapshot_paths)} snapshots to ensemble.")
    print(snapshot_paths)

    # --- 앙상블 평가 루프 ---
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    
    # 모든 모델의 예측 확률을 저장할 리스트
    # (모델 수, 샘플 수, 클래스 수) 형태가 됩니다.
    ensemble_probs = []

    # 1. 각 스냅샷 모델로 예측 확률 계산
    for snapshot_path in tqdm(snapshot_paths, desc="[Ensemble Inference]"):
        # 저장된 가중치 로드
        load_checkpoint(snapshot_path, model, device=config.DEVICE)
        
        batch_probs = []
        if not all_labels: # 레이블은 한 번만 저장하면 됩니다.
            is_first_model = True
        else:
            is_first_model = False

        with torch.no_grad():
            for motion_features, labels in val_loader:
                motion_features = motion_features.to(config.DEVICE)
                
                
                outputs = model(motion_features)
                
                # Softmax를 적용하여 확률 값으로 변환
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu())
                
                if is_first_model:
                    all_labels.append(labels.cpu())

        # 현재 모델의 모든 예측 확률을 하나로 합침
        ensemble_probs.append(torch.cat(batch_probs))

    # 2. 예측 확률을 평균내어 최종 예측 결정
    # 리스트를 텐서로 변환 (모델 수, 샘플 수, 클래스 수)
    ensemble_probs_tensor = torch.stack(ensemble_probs)
    # 모델 축(dim=0)에 대해 평균 계산
    avg_probs = torch.mean(ensemble_probs_tensor, dim=0) # (샘플 수, 클래스 수)
    
    # 평균 확률에서 가장 높은 값을 가진 클래스를 최종 예측으로 선택
    _, predicted_labels = torch.max(avg_probs, 1)
    
    ground_truth_labels = torch.cat(all_labels)
    
    # 3. 정확도 계산
    total_samples = len(ground_truth_labels)
    correct_predictions = (predicted_labels == ground_truth_labels).sum().item()

    # --- 최종 결과 출력 ---
    final_accuracy = correct_predictions / total_samples
    
    print("\n--- Ensemble Evaluation Finished ---")
    print(f"Ensembled {len(snapshot_paths)} models.")
    print(f"Final Accuracy: {final_accuracy * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a snapshot ensemble of models.")
    parser.add_argument('-d', '--checkpoint_dir', type=str, required=True,
                        help="Directory containing the snapshot checkpoint files (e.g., 'checkpoints/')")
    args = parser.parse_args()
    
    evaluate_ensemble(args.checkpoint_dir)
