import optuna
import subprocess
import sys
import re
import os
import pty  # [핵심] 가짜 터미널 라이브러리 추가

# ---------------------------------------------------------
# [설정] 탐색할 횟수와 스터디 이름
# ---------------------------------------------------------
N_TRIALS = 30          
STUDY_NAME = "finalStudy1"  
STORAGE_URL = f"sqlite:///{STUDY_NAME}.db" 

def objective(trial):
    # -----------------------------------------------------
    # 1. 하이퍼파라미터 정의
    # -----------------------------------------------------
    lr = trial.suggest_float("lr", 4e-4, 9e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    prob = trial.suggest_float("prob", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    smoothing = trial.suggest_float("smoothing", 0.05, 0.2)

    print(f"\n[Trial {trial.number}] Starting training...")
    print(f"Params: LR={lr:.6f}, DO={dropout:.3f}, Alpha={alpha:.3f}, Prob={prob:.3f}, WD={weight_decay:.5f}")

    python_executable = sys.executable 
    
    command = [
        python_executable, "train.py", # -u 옵션 없어도 됩니다. pty가 처리함.
        "--study-name", STUDY_NAME,
        "--trial-number", str(trial.number),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--alpha", str(alpha),
        "--prob", str(prob),
        "--weight-decay", str(weight_decay),
        "--smoothing", str(smoothing),
        "--protocol", "xsub",    
        "--scheduler", "cosine_decay"
    ]

    # -----------------------------------------------------
    # 2. [핵심 수정] pty를 사용하여 가짜 터미널 생성
    # -----------------------------------------------------
    master_fd, slave_fd = pty.openpty()
    
    try:
        # slave_fd를 subprocess의 stdout/stderr로 넘겨주면
        # train.py는 자기가 터미널에서 실행되는 줄 알고 tqdm을 정상 출력함
        process = subprocess.Popen(
            command,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True, # 자식 프로세스에서 불필요한 파일 디스크립터 닫기
            text=True
        )
        
        # 부모 프로세스에서는 slave_fd를 닫아줘야 함 (자식이 쓰니까)
        os.close(slave_fd)

        best_accuracy = 0.0
        full_output_log = ""

        # 3. master_fd를 통해 출력 읽기
        while True:
            try:
                # master_fd에서 데이터를 1024바이트씩 읽음
                output = os.read(master_fd, 1024).decode('utf-8', errors='replace')
            except OSError:
                # 프로세스가 끝나서 읽을 게 없으면 에러 발생 -> 루프 종료
                break

            if not output:
                break

            # (1) 화면에 즉시 출력 (줄바꿈 없이 그대로 쏴줌 -> tqdm 애니메이션 정상 작동)
            print(output, end='', flush=True)
            
            # (2) 정확도 파싱을 위해 로그 모으기
            full_output_log += output

        # 프로세스 종료 대기
        process.wait()

        # 4. 결과 파싱
        # 로그 전체에서 마지막 Best Accuracy 찾기
        matches = re.findall(r"Best Validation Accuracy: (\d+\.\d+)", full_output_log)
        if matches:
            best_accuracy = float(matches[-1]) # 가장 마지막에 찍힌 값 사용

        if process.returncode != 0:
            print(f"[Error] Trial {trial.number} failed with return code {process.returncode}")
            return 0.0

        print(f"\n[Trial {trial.number}] Finished. Captured Accuracy: {best_accuracy:.4f}")
        return best_accuracy

    except Exception as e:
        print(f"[Exception] {e}")
        # 혹시 모를 파일 디스크립터 정리
        try: os.close(master_fd)
        except: pass
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
        print("\n\n" + "="*40)
        print("  ✋ 사용자가 중단 요청을 보냈습니다.")
        print("  데이터베이스는 안전하게 저장되었습니다.")
        print("="*40 + "\n")
        try: sys.exit(0)
        except: pass

    print("\nOptimization Finished!")
    if len(study.trials) > 0:
        best_trial = study.best_trial
        print(f"Best Accuracy: {best_trial.value:.4f}")
        print("Best Params:", best_trial.params)
