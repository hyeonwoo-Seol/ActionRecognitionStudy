import torch
import time
import numpy as np
from ptflops import get_model_complexity_info
from model import SlowFast_Transformer
import config

def input_constructor(input_res):
    """
    ptflops가 모델에 입력을 넣을 때 사용하는 생성자 함수입니다.
    SlowFast_Transformer는 두 개의 입력(Fast, Slow)을 받으므로
    이를 튜플 형태로 반환해야 합니다.
    
    [중요 수정 사항]
    제공된 model.py 분석 결과, Slow 경로는 Fast 경로 대비 
    시간 축(T) 길이가 절반(1/2)이어야 합니다.
    """
    batch_size = 1
    
    # Fast 경로: 원본 프레임 수 (예: 100)
    T_fast = config.MAX_FRAMES
    
    # Slow 경로: 절반 프레임 수 (예: 50)
    # model.py의 Lateral Connection과 Positional Encoding이 T/2를 기준으로 설계됨
    T_slow = config.MAX_FRAMES // 2
    
    fast_shape = (batch_size, config.NUM_COORDS, T_fast, config.NUM_JOINTS)
    slow_shape = (batch_size, config.NUM_COORDS, T_slow, config.NUM_JOINTS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_fast = torch.randn(fast_shape).to(device)
    dummy_slow = torch.randn(slow_shape).to(device)
    
    # 모델의 forward(*args)에 들어갈 인자들을 딕셔너리(키워드 인자)로 반환
    return {"x_fast": dummy_fast, "x_slow": dummy_slow}

def measure_efficiency():
    # 1. 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # config.NUM_SUBJECTS 등은 config.py에서 자동으로 가져옵니다.
    model = SlowFast_Transformer().to(device)
    model.eval()

    # ----------------------------------------------------------------------
    # 2. FLOPs 및 Parameters 측정 (ptflops 사용)
    # ----------------------------------------------------------------------
    print("\n[1] Measuring FLOPs & Params with ptflops...")
    
    try:
        # input_res는 input_constructor를 사용할 경우 무시되지만 형식상 넣어줍니다.
        macs, params = get_model_complexity_info(
            model, 
            (0,), # dummy input resolution (ignored due to input_constructor)
            as_strings=False, 
            print_per_layer_stat=False, 
            verbose=False,
            input_constructor=input_constructor
        )
        
        # ptflops는 기본적으로 MACs를 반환하므로 FLOPs로 변환 (x2)
        flops = macs * 2

        print(f" - Parameters : {params / 1e6:.2f} M")
        print(f" - MACs       : {macs / 1e9:.2f} G")
        print(f" - FLOPs      : {flops / 1e9:.2f} G (Calculated as MACs * 2)")
        
    except Exception as e:
        print(f"Error measuring FLOPs: {e}")
        print("입력 텐서의 차원(Fast/Slow)이 모델 설정과 일치하는지 확인해주세요.")
        return

    # ----------------------------------------------------------------------
    # 3. Latency 및 FPS 측정
    # ----------------------------------------------------------------------
    print("\n[2] Measuring Latency & FPS...")
    
    # 더미 입력 생성 (Batch Size = 1)
    T_fast = config.MAX_FRAMES
    T_slow = config.MAX_FRAMES // 2
    
    dummy_fast = torch.randn(1, config.NUM_COORDS, T_fast, config.NUM_JOINTS).to(device)
    dummy_slow = torch.randn(1, config.NUM_COORDS, T_slow, config.NUM_JOINTS).to(device)

    # GPU Warm-up (초기 오버헤드 제거)
    warmup_iters = 50
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_fast, dummy_slow)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 실제 측정
    test_iters = 1000
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(dummy_fast, dummy_slow)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = total_time / test_iters # 초 단위
    fps = 1.0 / avg_latency

    print(f" - Batch Size : 1")
    print(f" - Latency    : {avg_latency * 1000:.2f} ms")
    print(f" - FPS        : {fps:.2f} frames/sec")

    print("\n" + "="*40)
    print(f"Summary for Report:")
    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs : {flops / 1e9:.2f} G (ptflops 기준)")
    print(f"Time  : {avg_latency * 1000:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    measure_efficiency()
