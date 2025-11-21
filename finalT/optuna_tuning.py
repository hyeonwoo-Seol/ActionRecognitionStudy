import optuna
import subprocess
import sys
import re
import os

# ---------------------------------------------------------
# [설정] 탐색할 횟수와 스터디 이름
# ---------------------------------------------------------
N_TRIALS = 20          # 총 시도할 실험 횟수
STUDY_NAME = "finalStudy1"  # 스터디 이름
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
    prob = trial.suggest_float("prob", 0.1, 0.5)
    
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
        python_executable, "-u", "train.py",
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
    best_accuracy = 0.0
    
    try:
        # Popen을 사용하여 프로세스를 열고 실시간으로 출력을 읽습니다.
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, # 에러도 표준출력으로 합침
            text=True, 
            encoding='utf-8',
            bufsize=1 # 라인 버퍼링
        )
        
        # 한 줄씩 읽어서 화면에 출력하고, 정확도를 파싱합니다.
        for line in process.stdout:
            print(line, end='') # 터미널에 즉시 출력
            
            # 정확도 파싱
            match = re.search(r"Best Validation Accuracy: (\d+\.\d+)", line)
            if match:
                best_accuracy = float(match.group(1))
        
        # 프로세스가 끝날 때까지 대기
        process.wait()

        if process.returncode != 0:
            print(f"[Error] Trial {trial.number} failed with return code {process.returncode}")
            return 0.0

        print(f"[Trial {trial.number}] Finished. Captured Accuracy: {best_accuracy:.4f}")
        return best_accuracy

    except Exception as e:
        print(f"[Exception] {e}")
        return 0.0

if __name__ == "__main__":
    print(f"Optuna Study: {STUDY_NAME}")
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        load_if_exists=True
    )

    print("Starting optimization...")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\nOptimization Finished!")
    best_trial = study.best_trial
    print(f"Best Accuracy: {best_trial.value:.4f}")
    print("Best Params:", best_trial.params)
