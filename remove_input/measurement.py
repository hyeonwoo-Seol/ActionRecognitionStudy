import torch
import time
import numpy as np
import config
from model import SlowFast_Transformer
from thop import profile

def measure_model_performance():
    # ----------------------------------------------------------------------
    # 1. 설정 (Edge Device 환경 가정)
    # ----------------------------------------------------------------------
    # 엣지 디바이스에서는 보통 배치 사이즈를 1로 두고 실시간 처리를 합니다.
    TEST_BATCH_SIZE = 1
    
    # 로봇 환경에 맞는 프레임 수 설정
    TEST_FRAMES = config.MAX_FRAMES 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Environment] Device: {device}")
    print(f"[Settings] Batch: {TEST_BATCH_SIZE}, Frames: {TEST_FRAMES}, Joints: {config.NUM_JOINTS}")

    # ----------------------------------------------------------------------
    # 2. 모델 로드 (SlowFast Transformer)
    # ----------------------------------------------------------------------
    # config.py 및 model.py의 정의에 맞춰 인자 수정
    model = SlowFast_Transformer(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        fast_dims=config.FAST_DIMS,
        slow_dims=config.SLOW_DIMS,
        num_aux_classes=config.NUM_SUBJECTS, # 측정용이므로 임의의 값(Subject 수) 전달
        alpha=0.0 # 측정 시 GRL Alpha는 영향 없음
    ).to(device)
    model.eval()

    # ----------------------------------------------------------------------
    # 3. 더미 입력 데이터 생성 (Fast & Slow Pathway)
    # ----------------------------------------------------------------------
    # ntu_data_loader.py를 참고하면 입력 텐서의 형태는 (N, C, T, J) 입니다.
    # Slow 경로는 Fast 경로의 프레임을 1/2로 다운샘플링한 것입니다.
    
    # Fast Pathway Input: (1, 12, T, 50)
    dummy_fast = torch.randn(
        TEST_BATCH_SIZE, 
        config.NUM_COORDS, 
        TEST_FRAMES, 
        config.NUM_JOINTS
    ).to(device)

    # Slow Pathway Input: (1, 12, T/2, 50)
    dummy_slow = torch.randn(
        TEST_BATCH_SIZE, 
        config.NUM_COORDS, 
        TEST_FRAMES // 2, 
        config.NUM_JOINTS
    ).to(device)

    print(f"[Inputs] Fast: {dummy_fast.shape}, Slow: {dummy_slow.shape}")

    # ----------------------------------------------------------------------
    # 4. Parameters (모델 크기) 측정
    # ----------------------------------------------------------------------
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[1] Model Size")
    print(f"    - Parameters: {num_params / 1e6:.2f} M (Million)")
    
    # ----------------------------------------------------------------------
    # 5. FLOPs (연산량) 측정
    # ----------------------------------------------------------------------
    # thop.profile에 두 개의 입력을 튜플로 전달
    try:
        macs, params = profile(model, inputs=(dummy_fast, dummy_slow), verbose=False)
        flops = macs * 2 # 통상적으로 1 MAC = 2 FLOPs로 계산
        
        print(f"\n[2] Computational Cost")
        print(f"    - MACs (Multiply-Accumulates): {macs / 1e9:.3f} G (Giga)")
        print(f"    - FLOPs (Floating Point Ops) : {flops / 1e9:.3f} G (Giga)")
        print(f"    * Note: 젯슨 나노급은 보통 1~5 GFLOPs 이내 권장")
    except Exception as e:
        print(f"\n[2] Computational Cost calculation failed: {e}")
        print("    (thop 라이브러리 호환성 문제일 수 있음)")

    # ----------------------------------------------------------------------
    # 6. Latency (지연 시간) 및 FPS 측정
    # ----------------------------------------------------------------------
    print(f"\n[3] Latency & FPS Measurement (Warmup + 100 runs)")
    
    # GPU Warmup (초기 캐싱으로 인한 지연 제거)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_fast, dummy_slow)
    
    # 시간 측정
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    
    if torch.cuda.is_available():
        # 동기화 객체 (GPU 시간 측정 정확도를 위함)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_fast, dummy_slow)
                ender.record()
                
                # GPU 연산이 끝날 때까지 대기
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) # 밀리초(ms) 단위 반환
                timings[rep] = curr_time
    else:
        # CPU 측정
        with torch.no_grad():
            for rep in range(repetitions):
                start_time = time.time()
                _ = model(dummy_fast, dummy_slow)
                end_time = time.time()
                timings[rep] = (end_time - start_time) * 1000 # ms 단위 변환

    avg_latency_ms = np.mean(timings)
    std_latency_ms = np.std(timings)
    fps = 1000 / avg_latency_ms
    
    print(f"    - Avg Latency: {avg_latency_ms:.4f} ms (+/- {std_latency_ms:.4f})")
    print(f"    - Throughput : {fps:.2f} FPS")
    
    # ----------------------------------------------------------------------
    # 7. Peak Memory (최대 메모리 사용량)
    # ----------------------------------------------------------------------
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        print(f"\n[4] Peak Memory Usage (Inference)")
        print(f"    - VRAM: {max_mem:.2f} MB")

    # ----------------------------------------------------------------------
    # 결과 해석 가이드
    # ----------------------------------------------------------------------
    print("\n" + "="*50)
    print(" [ Robot / Edge Device Suitability Check ]")
    print("="*50)
    
    # 1. 파라미터 기준
    if num_params < 5e6: msg_p = "Excellent (Very Light)"
    elif num_params < 10e6: msg_p = "Good (Fit for Mobile)"
    else: msg_p = "Heavy (Might need optimization)"
    print(f" 1. Size     : {msg_p}")
    
    # 2. FPS 기준 (일반적인 로봇 제어 루프 기준)
    if fps >= 30: msg_f = "Perfect (Real-time)"
    elif fps >= 15: msg_f = "Good (Usable)"
    else: msg_f = "Slow (Might feel laggy)"
    print(f" 2. Speed    : {msg_f}")
    
    print("="*50)

if __name__ == "__main__":
    measure_model_performance()
