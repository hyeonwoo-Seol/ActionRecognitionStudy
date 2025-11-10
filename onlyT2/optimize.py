# optimize.py

"""
Optuna HPO(Hyperparameter Optimization) 스터디를 실행합니다.

이 스크립트는 'study.optimize()' 워크플로우를 사용하여
지정된 'n_trials' 횟수만큼 HPO를 자동으로 수행합니다.

'train.py'의 'run_trial' 함수를 호출하여 각 Trial을 실행하고,
Pruning(조기 종료)을 활성화하여 비효율적인 Trial을 조기에 중단시킵니다.

[실행 방법]
# 1. 'xsub' 프로토콜로 'slowfast_tuning_v2' 스터디 100회 실행 (기본값)
python optimize.py --study-name slowfast_tuning_v2 --n-trials 100

# 2. 'xview' 프로토콜로 실행
python optimize.py --study-name slowfast_tuning_xview --n-trials 50 --protocol xview
"""

import optuna
import argparse
import sys
from argparse import Namespace # args 객체를 만들기 위해 import

# 1. [요구사항] train.py에서 run_trial 함수를 import
from train import run_trial
# config.py는 Pruner의 n_warmup_steps를 참조하기 위해 import
import config


# 2. [요구사항] objective(trial) 함수 정의
def objective(trial, cli_args):
    """Optuna가 단일 Trial을 평가하기 위해 호출하는 함수"""
    
    # 3. [요구사항] argparse.Namespace를 사용하여 args 객체 생성
    args = Namespace()

    # 4. [요구사항] manager.py의 파라미터를 trial.suggest_...로 제안
    # (이름은 train.py의 argparse와 일치시킴: lr, dropout 등)
    args.lr = trial.suggest_float("LEARNING_RATE", 1e-4, 5e-4, log=True)
    args.dropout = trial.suggest_float("DROPOUT", 0.2, 0.5)
    args.alpha = trial.suggest_float("ADVERSARIAL_ALPHA", 0.05, 0.3)
    args.prob = trial.suggest_float("PROB", 0.3, 0.7)
    args.weight_decay = trial.suggest_float("ADAMW_WEIGHT_DECAY", 0.01, 0.1, log=True)
    args.smoothing = trial.suggest_float("LABEL_SMOOTHING", 0.0, 0.15)

    # 5. HPO 대상이 아닌, 고정 인자들 설정
    # (cli_args에서 --scheduler, --protocol 등을 가져옴)
    args.scheduler = cli_args.scheduler
    args.protocol = cli_args.protocol
    args.study_name = cli_args.study_name
    
    # 6. [요구사항] trial 객체에서 trial_number를 가져와 args에 추가
    # (train.py의 run_trial이 이 값을 사용하여 체크포인트 폴더를 만듭니다)
    args.trial_number = trial.number

    # 7. [요구사항] run_trial(args, trial) 형태로 호출
    # run_trial 함수는 학습 완료 후 'best_accuracy'를 반환합니다.
    try:
        best_accuracy = run_trial(args, trial)
        return best_accuracy
    
    except Exception as e:
        # run_trial에서 예상치 못한 오류 발생 시 (예: OOM)
        # Optuna가 이 Trial을 'FAIL'로 기록하도록 예외를 다시 발생시킵니다.
        print(f"\n--- [Error in Trial {trial.number}] ---")
        print(f"An unexpected error occurred: {e}")
        # (KeyboardInterrupt는 train.py 내부에서 처리되므로 여기에 도달하지 않습니다)
        raise e # Optuna가 예외를 잡도록 다시 발생


# --- 메인 실행 블록 ---
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run Optuna HPO study.")
    
    # --- 1. 스터디 설정 인자 (manager.py에서 가져옴) ---
    parser.add_argument(
        '--study-name', 
        type=str, 
        default="slowfast_tuning_v2",
        help="Name for the Optuna study DB file (e.g., 'study_v2.db')"
    )
    parser.add_argument(
        '--n-trials', 
        type=int, 
        default=100,
        help="Number of trials to run in this optimization session."
    )

    # --- 2. train.py에 전달할 고정 인자 (train.py 참조) ---
    parser.add_argument(
        '--scheduler', 
        type=str, 
        default='cosine_decay', 
        choices=['cosine_decay', 'cosine_restarts'],
        help="Scheduler to use: 'cosine_decay' or 'cosine_restarts'"
    )
    parser.add_argument(
        '-p', '--protocol', 
        type=str, 
        default='xsub', 
        choices=['xsub', 'xview'],
        help="Training protocol: 'xsub' (Cross-Subject) or 'xview' (Cross-View)."
    )
    
    cli_args = parser.parse_args()

    # --- 3. Pruner 설정 (Pruning 활성화) ---
    # config.py의 WARMUP_EPOCHS 값을 기준으로 설정합니다.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # 최소 5개의 Trial은 Pruning 없이 끝까지 실행
        n_warmup_steps=config.WARMUP_EPOCHS, # 최소 7 에포크는 Pruning 안 함
        interval_steps=1 # 7 에포크 이후 매 에포크마다 Pruning 여부 검사
    )
    print(f"Pruner enabled: MedianPruner(n_warmup_steps={config.WARMUP_EPOCHS})")

    # --- 4. 스터디 생성 (manager.py 로직) ---
    storage_name = f"sqlite:///{cli_args.study_name}.db"
    
    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=storage_name,
        direction='maximize',   # 정확도(Accuracy)를 '최대화'
        load_if_exists=True,    # DB 파일이 있으면 이어서 실행
        pruner=pruner           # Pruner 적용
    )

    print(f"Study '{cli_args.study_name}' loaded/created.")
    print(f"Storage: {storage_name}")
    print(f"Running {cli_args.n_trials} trials...")

    # --- 5. [요구사항] study.optimize() 호출 ---
    try:
        study.optimize(
            # lambda를 사용해 objective 함수에 고정 인자(cli_args) 전달
            lambda trial: objective(trial, cli_args),
            n_trials=cli_args.n_trials,
            n_jobs=1 # [중요] 단일 GPU 사용 시 1로 고정
        )
    except KeyboardInterrupt:
        print("\n--- [Study Interrupted] ---")
        print("Optuna study optimization was interrupted by user (Ctrl+C).")
        print("The current trial (if any) should have saved its 'resume_checkpoint.pth.tar'.")
        print(f"Run 'python {sys.argv[0]} --study-name {cli_args.study_name}' to resume.")
    except Exception as e:
        print(f"\n--- [Study Failed] ---")
        print(f"An critical error occurred during study.optimize: {e}")
        import traceback
        traceback.print_exc()

    # --- 6. 최종 결과 요약 ---
    print("\n" + "="*50)
    print("--- Study Optimization Finished ---")
    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    
    try:
        best_trial = study.best_trial
        print("\n--- Best Trial ---")
        print(f"  Trial Number: {best_trial.number}")
        print(f"  Best Value (Accuracy): {best_trial.value:.4f}")
        print("  Best Hyperparameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("\nNo trials were completed successfully.")
        
    print("="*50)
    print(f"To view the dashboard, run:")
    print(f"optuna-dashboard {storage_name}")
