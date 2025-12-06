import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis
from model import SlowFast_Transformer
import config

def compare_flops_measurement():
    # 1. 모델 및 데이터 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SlowFast_Transformer().to(device)
    model.eval()

    # (Batch=1) 입력 데이터 생성
    # config에 맞게 차원 설정 (채널, 프레임, 조인트 등)
    dummy_fast = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)
    dummy_slow = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES // 2, config.NUM_JOINTS).to(device)
    inputs = (dummy_fast, dummy_slow)

    print(f"--- FLOPs Measurement Comparison ---")

    # ---------------------------------------------------
    # Method 1: thop
    # ---------------------------------------------------
    macs_thop, params_thop = profile(model, inputs=inputs, verbose=False)
    flops_thop = macs_thop * 2 # MACs -> FLOPs 변환
    print(f"[thop]")
    print(f" - Parameters: {params_thop / 1e6:.2f} M")
    print(f" - GFLOPs    : {flops_thop / 1e9:.2f} G")

    # ---------------------------------------------------
    # Method 2: fvcore (순서 수정됨)
    # ---------------------------------------------------
    flops_analyzer = FlopCountAnalysis(model, inputs)
    
    # [수정] 먼저 계산(.total)을 수행해야 합니다.
    flops_fvcore_raw = flops_analyzer.total()
    
    # 그 다음 지원하지 않는 연산이 있었는지 확인합니다.
    unsupported = flops_analyzer.unsupported_ops()
    if unsupported:
        print(f" * Warning: fvcore found unsupported ops: {unsupported}")

    # fvcore 결과 변환 (보통 1 MAC 단위이므로 2를 곱해 비교)
    final_flops_fvcore = flops_fvcore_raw * 2
    
    print(f"[fvcore]")
    print(f" - GFLOPs    : {final_flops_fvcore / 1e9:.2f} G")
    
    # ---------------------------------------------------
    # 비교 결론
    # ---------------------------------------------------
    diff = abs(flops_thop - final_flops_fvcore)
    print(f"\n[Conclusion]")
    print(f"Difference: {diff / 1e9:.4f} G")
    
    # 오차율 계산
    if flops_thop > 0:
        error_rate = diff / flops_thop
        if error_rate < 0.05: # 5% 미만 차이
            print(">> 두 도구의 결과가 거의 일치합니다. (thop 신뢰 가능)")
        else:
            print(f">> 차이가 있습니다 (오차율 {error_rate*100:.1f}%).")
            print("   fvcore 경고 메시지(unsupported ops)를 확인해보세요.")

if __name__ == "__main__":
    compare_flops_measurement()
