import torch
import numpy as np
import time
from thop import profile  # pip install thop
from model import SlowFast_Transformer
import config

def measure_performance():
    # 1. 장치 설정 및 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Stats measure on: {device}")
    
    model = SlowFast_Transformer().to(device)
    model.eval()

    # 2. 더미 입력 데이터 생성 (Batch Size=1)
    # config에 정의된 입력 크기: (1, 12, 100, 50) 및 (1, 12, 50, 50)
    C = config.NUM_COORDS
    T = config.MAX_FRAMES
    J = config.NUM_JOINTS
    
    # Fast Path Input
    dummy_fast = torch.randn(1, C, T, J).to(device)
    # Slow Path Input (T/2)
    dummy_slow = torch.randn(1, C, T // 2, J).to(device)

    # ---------------------------------------------------------
    # 3. Parameters & FLOPs 측정 (thop 라이브러리 사용)
    # ---------------------------------------------------------
    # inputs는 튜플로 전달해야 함: (fast, slow)
    macs, params = profile(model, inputs=(dummy_fast, dummy_slow), verbose=False)
    
    flops = macs * 2  # 통상적으로 1 MAC = 2 FLOPs
    print(f"\n[Model Complexity]")
    print(f" - Parameters : {params / 1e6:.2f} M")  # Million 단위
    print(f" - FLOPs      : {flops / 1e9:.2f} G")   # Giga 단위

    # ---------------------------------------------------------
    # 4. Throughput (FPS) & Latency 측정
    # ---------------------------------------------------------
    print(f"\n[Inference Speed]")
    
    # (1) Warm-up: 초기화 오버헤드 제거를 위해 미리 실행
    print(" - Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_fast, dummy_slow)
    
    # (2) 실제 측정
    repetitions = 1000  # 측정 반복 횟수
    timings = []       # 시간 기록용 리스트

    # GPU 타이밍을 위한 이벤트 설정
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    print(f" - Measuring latency over {repetitions} runs...")
    
    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model(dummy_fast, dummy_slow)
            ender.record()
            
            # GPU 동기화 (필수)
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 밀리초(ms) 단위 반환
            timings.append(curr_time)

    # (3) 결과 계산
    avg_latency = np.mean(timings)
    std_latency = np.std(timings)
    fps = 1000 / avg_latency  # 1000ms / 평균시간ms
    
    print(f" - Avg Latency: {avg_latency:.4f} ms (±{std_latency:.2f})")
    print(f" - FPS        : {fps:.2f} frames/sec")

if __name__ == "__main__":
    measure_performance()
