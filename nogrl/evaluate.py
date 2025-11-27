# evaluate.py
# ## --------------------------------------------------------------------
# GRL 제거 버전 평가 스크립트
# ## --------------------------------------------------------------------

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.amp import autocast
import numpy as np
import os
from thop import profile
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import config
from ntu_data_loader import NTURGBDDataset, DataLoader
from model import SlowFast_Transformer 
from utils import load_checkpoint

def calculate_topk_accuracy(outputs, labels, k=5):
    with torch.no_grad():
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()

def generate_tsne_plot(features_list, action_labels_list, domain_labels_list, protocol):
    print("\n--- Generating t-SNE Visualization ---")
    
    all_features = torch.cat(features_list, dim=0).numpy()
    all_action_labels = torch.cat(action_labels_list, dim=0).numpy()
    all_domain_labels = torch.cat(domain_labels_list, dim=0).numpy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=config.SEED, n_jobs=-1)
    features_2d = tsne.fit_transform(features_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    num_classes = config.NUM_CLASSES
    
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1], hue=all_action_labels,
        palette=sns.color_palette("hsv", num_classes), s=10, alpha=0.7, legend=None, ax=ax1
    )
    ax1.set_title(f't-SNE colored by Action Labels (K={num_classes})', fontsize=16)

    # [수정됨] GRL은 없지만 도메인 분포 확인용으로 남겨둘 수 있음
    domain_names = {0: 'Source Domain (Train)', 1: 'Target Domain (Val)'}
    domain_labels_named = [domain_names[label] for label in all_domain_labels]
    
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1], hue=domain_labels_named,
        palette=['blue', 'orange'], s=10, alpha=0.7, legend="full", ax=ax2
    )
    ax2.set_title(f't-SNE colored by Domain ({protocol}) - No GRL', fontsize=16)

    save_path = f"tsne_plot_{protocol}_no_grl.png"
    plt.savefig(save_path, dpi=150)
    print(f"t-SNE visualization saved to '{save_path}'")
    plt.close()

def calculate_flops(model, device):
    print("\n--- Calculating FLOPs (using thop) ---")
    model.eval() 
    T_fast = config.MAX_FRAMES
    T_slow = config.MAX_FRAMES // 2 
    sample_fast = torch.randn(1, config.NUM_COORDS, T_fast, config.NUM_JOINTS).to(device)
    sample_slow = torch.randn(1, config.NUM_COORDS, T_slow, config.NUM_JOINTS).to(device)
    inputs = (sample_fast, sample_slow) 
    try:
        total_ops, total_params = profile(model, inputs=inputs, verbose=False)
        gflops = (total_ops * 2) / 1e9
        print(f"Model GFLOPs: {gflops:.2f} G")
    except Exception as e:
        print(f"Error during FLOPs calculation: {e}")
    print("---------------------------\n")

def calculate_params(model):
    print("--- Calculating Params ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print("---------------------------\n")

def evaluate_model(checkpoint_path, protocol, run_tsne):
    print(f"--- [{protocol.upper()}] Evaluation Mode (No GRL) ---")
    split_name = 'val'
    device = config.DEVICE
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'optuna_params' in checkpoint:
        for key, value in checkpoint['optuna_params'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    val_dataset = NTURGBDDataset(
        data_path=config.DATASET_PATH, split=split_name, max_frames=config.MAX_FRAMES, protocol=protocol
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )

    # [GRL 제거됨] alpha 인자 제거
    model = SlowFast_Transformer(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        fast_dims=config.FAST_DIMS,
        slow_dims=config.SLOW_DIMS,
        num_subjects=config.NUM_SUBJECTS
    ).to(device)

    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model weights loaded from '{checkpoint_path}'")

    calculate_params(model)
    calculate_flops(model, device)
    
    criterion = nn.CrossEntropyLoss()

    all_features_list = []
    all_action_labels_list = []
    all_domain_labels_list = []

    def hook_fn(module, input, output):
        # [수정됨] GRL이 없으므로 action_head의 입력(input[0])을 낚아챔
        all_features_list.append(input[0].detach().cpu())

    if run_tsne:
        print("Registering t-SNE hook on 'model.action_head' layer...")
        # [수정됨] model.grad_reversal 대신 model.action_head에 훅 등록
        model.action_head.register_forward_hook(hook_fn)

    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0 
    total_samples = 0
    
    eval_bar = tqdm(val_loader, desc=f"[Evaluate {protocol.upper()}]", colour="yellow")
    with torch.no_grad():
        for data_fast, data_slow, action_labels, subject_labels in eval_bar:
            
            data_fast = data_fast.to(device)
            data_slow = data_slow.to(device)
            action_labels = action_labels.to(device)
            
            with autocast(device_type=device):
                # [수정됨] outputs_action만 반환됨
                outputs_action = model(data_fast, data_slow)
                loss = criterion(outputs_action, action_labels)
            
            total_loss += loss.item() * data_fast.size(0)
            total_samples += action_labels.size(0)
            
            _, predicted_action = torch.max(outputs_action.data, 1)
            correct_top1 += (predicted_action == action_labels).sum().item()
            correct_top5 += calculate_topk_accuracy(outputs_action, action_labels, k=5)

            if run_tsne:
                all_action_labels_list.append(action_labels.cpu())
                if protocol == 'xview':
                    all_domain_labels_list.append(subject_labels.cpu())
                else: 
                    domain_labels = torch.ones_like(action_labels, dtype=torch.int)
                    all_domain_labels_list.append(domain_labels.cpu())

            avg_loss = total_loss / total_samples
            avg_acc_t1 = correct_top1 / total_samples
            
            # [수정됨] Subject Acc 제거
            eval_bar.set_postfix(
                loss=f"{avg_loss:.4f}", 
                acc_T1=f"{avg_acc_t1:.4f}"
            )

    final_loss = total_loss / total_samples
    final_acc_top1 = correct_top1 / total_samples
    final_acc_top5 = correct_top5 / total_samples
    
    print(f"\n--- Evaluation Finished ({protocol.upper()}) ---")
    print(f"Average Loss: {final_loss:.4f}")
    print(f"Top-1 Action Accuracy: {final_acc_top1 * 100:.2f}%")
    print(f"Top-5 Action Accuracy: {final_acc_top5 * 100:.2f}%")
    # [GRL 제거됨] Subject Acc 출력 제거

    if run_tsne:
        generate_tsne_plot(
            all_features_list, 
            all_action_labels_list, 
            all_domain_labels_list,
            protocol
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the SlowFast Transformer model.")
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument('-p', '--protocol', type=str, default='xsub', choices=['xsub', 'xview'],
                        help="Evaluation protocol")
    parser.add_argument('--tsne', action='store_true',
                        help="Run t-SNE visualization")
    
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.protocol, args.tsne)
