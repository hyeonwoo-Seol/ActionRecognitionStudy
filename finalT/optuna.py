import optuna
import subprocess
import sys
import re
import os

# ---------------------------------------------------------
# [설정] 탐색할 횟수와 스터디 이름
# ---------------------------------------------------------
N_TRIALS = 30          # 총 시도할 실험 횟수
STUDY_NAME = "slowfast_optimization"  # 스터디 이름
STORAGE_URL = f"sqlite:///{STUDY_NAME}.db" # 결과를 저장할 DB 파일 (중단 후 재개 가능)

def objective(trial):
    """
    Optuna가 실행할 목적 함수입니다.
    1. 하이퍼파라미터를 추천(Suggest)받습니다.
    2. train.py를 서브프로세스로 실행합니다.
    3. 결과(정확도)를 파싱하여 반환합니다.
    """
    
    # -----------------------------------------------------
    # 1. 하이퍼파라미터 탐색 공간 정의
    # -----------------------------------------------------
    # (1) 학습률: 로그 스케일로 탐색
    lr = trial.suggest_float("lr", 4e-4, 5e-4, log=True)
    
    # (2) 드롭아웃: 0.2 ~ 0.6 사이
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    
    # (3) GRL Alpha (적대적 학습 강도)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    
    # (4) 데이터 증강 스킵 확률
    prob = trial.suggest_float("prob", 0.5, 0.1)
    
    # (5) 가중치 감쇠 (Weight Decay): 로그 스케일
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    
    # (6) Label Smoothing: 0.0 ~ 0.3
    smoothing = trial.suggest_float("smoothing", 0.05, 0.2)

    
    # -----------------------------------------------------
    # 2. train.py 실행 명령어 생성
    # -----------------------------------------------------
    print(f"\n[Trial {trial.number}] Starting training with params:")
    print(f"LR={lr:.6f}, DO={dropout:.3f}, Alpha={alpha:.3f}, Prob={prob:.3f}, WD={weight_decay:.5f}")

    # 현재 파이썬 인터프리터 경로 (가상환경 호환성 위해)
    python_executable = sys.executable 
    
    command = [
        python_executable, "train.py",
        "--study-name", STUDY_NAME,
        "--trial-number", str(trial.number),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--alpha", str(alpha),
        "--prob", str(prob),
        "--weight-decay", str(weight_decay),
        "--smoothing", str(smoothing),
        "--protocol", "xsub",     # 또는 'xview'
        "--scheduler", "cosine_decay"
    ]

    # -----------------------------------------------------
    # 3. 프로세스 실행 및 로그 캡처
    # -----------------------------------------------------
    try:
        # train.py 실행 (stdout을 파이프로 캡처)
        # text=True로 설정하여 문자열로 바로 받음
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        
        # 실행 중 오류가 발생했는지 확인
        if result.returncode != 0:
            print(f"[Error] Trial {trial.number} failed.")
            print("--- Stderr ---")
            print(result.stderr)
            return 0.0 # 실패 시 점수 0점 처리

        # -------------------------------------------------
        # 4. 결과 파싱 (Best Validation Accuracy 찾기)
        # -------------------------------------------------
        output_log = result.stdout
        
        # train.py가 마지막에 출력하는 "Best Validation Accuracy: 0.xxxx" 패턴 찾기
        match = re.search(r"Best Validation Accuracy: (\d+\.\d+)", output_log)
        
        if match:
            accuracy = float(match.group(1))
            print(f"[Trial {trial.number}] Finished. Accuracy: {accuracy:.4f}")
            return accuracy
        else:
            print(f"[Warning] Could not parse accuracy from Trial {trial.number}.")
            # 로그의 마지막 부분 일부 출력 (디버깅용)
            print("--- Last 500 chars of output ---")
            print(output_log[-500:])
            return 0.0

    except Exception as e:
        print(f"[Exception] {e}")
        return 0.0

if __name__ == "__main__":
    # 1. 저장소(DB)가 있으면 불러오고, 없으면 새로 생성
    print(f"Optuna Study: {STUDY_NAME}")
    print(f"Storage: {STORAGE_URL}")
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize", # 정확도는 높을수록 좋음
        load_if_exists=True   # 기존 기록이 있다면 이어서 진행
    )

    # 2. 최적화 실행
    print("Starting optimization...")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    # 3. 최종 결과 리포트
    print("\n" + "="*50)
    print("Optimization Finished!")
    print("="*50)
    
    best_trial = study.best_trial
    print(f"Best Trial ID: {best_trial.number}")
    print(f"Best Accuracy: {best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
