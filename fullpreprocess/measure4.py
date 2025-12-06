import torch
import numpy as np
import config
from model import SlowFast_Transformer

def measure_inference_speed():
    # ------------------------------------------------------------------------
    # 1. 환경 설정 (Setup)
    # ------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("Warning: CPU 측정은 정확한 FPS 비교가 어렵습니다. GPU 사용을 권장합니다.")

    # 모델 로드 (Evaluation 모드 필수)
    model = SlowFast_Transformer().to(device)
    model.eval()

    # ------------------------------------------------------------------------
    # 2. 더미 데이터 준비 (Batch Size = 1)
    # ------------------------------------------------------------------------
    # Fast Path: (1, 12, 100, 50)
    dummy_fast = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)
    # Slow Path: (1, 12, 50, 50) - 프레임 절반
    dummy_slow = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES // 2, config.NUM_JOINTS).to(device)

    # ------------------------------------------------------------------------
    # 3. 웜업 (Warm-up)
    # ------------------------------------------------------------------------
    # GPU 초기화 및 캐싱 등으로 인한 첫 실행 딜레이를 제거하기 위함
    print("warming up...", end=" ")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_fast, dummy_slow)
    print("Done.")

    # ------------------------------------------------------------------------
    # 4. 실제 측정 (Measurement) - torch.cuda.Event 사용
    # ------------------------------------------------------------------------
    repetitions = 1000  # 반복 횟수 (많을수록 오차가 줄어듦)
    timings = []        # 측정된 시간(ms) 저장

    # GPU 타이머 이벤트 생성
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    print(f"Measuring latency over {repetitions} runs...")
    
    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()   # 시작 시간 기록
            
            # 모델 추론
            _ = model(dummy_fast, dummy_slow)
            
            ender.record()     # 종료 시간 기록
            
            # [중요] GPU 연산이 완전히 끝날 때까지 대기 (동기화)
            torch.cuda.synchronize()
            
            # 시간 계산 (밀리초 단위)
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

    # ------------------------------------------------------------------------
    # 5. 결과 계산 (Metrics)
    # ------------------------------------------------------------------------
    # Latency: 추론 한 번에 걸리는 시간
    mean_latency = np.mean(timings)
    std_latency = np.std(timings)
    
    # FPS (Frames Per Second): 1초(1000ms)에 몇 번 추론 가능한가?
    # 주의: 여기서 Frame은 '비디오 프레임'이 아니라 '입력 배치(Input Sample)' 단위입니다.
    # 즉, "이 모델은 1초에 100개의 비디오 클립을 처리할 수 있다"는 뜻입니다.
    fps = 1000 / mean_latency

    print("\n" + "="*40)
    print(f" Model Speed Benchmark (Batch Size=1)")
    print("="*40)
    print(f" Latency : {mean_latency:.4f} ms ± {std_latency:.2f}")
    print(f" FPS     : {fps:.2f} (samples/sec)")
    print("="*40)

if __name__ == "__main__":
    measure_inference_speed()
