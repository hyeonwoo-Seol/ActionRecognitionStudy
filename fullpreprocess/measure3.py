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
    # [수정] 모델 내부에서 시간 축 압축(Embedding)을 수행하므로,
    # 입력은 Fast/Slow 모두 원본 길이(MAX_FRAMES)를 가집니다.
    # Shape: (Batch, Channels, Frames, Joints)
    dummy_fast = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)
    dummy_slow = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)
    
    inputs = (dummy_fast, dummy_slow)

    print(f"--- FLOPs Measurement Comparison ---")
    print(f"Input Shape: (N, C, T, V)")
    print(f" - Fast: {dummy_fast.shape}")
    print(f" - Slow: {dummy_slow.shape}")

    # ---------------------------------------------------
    # Method 1: thop
    # ---------------------------------------------------
    # verbose=False로 설정하여 불필요한 로그 출력 방지
    macs_thop, params_thop = profile(model, inputs=inputs, verbose=False)
    flops_thop = macs_thop * 2 # MACs -> FLOPs 변환 (일반적으로 1 MAC = 2 FLOPs)
    
    print(f"\n[thop]")
    print(f" - Parameters: {params_thop / 1e6:.2f} M")
    print(f" - GFLOPs    : {flops_thop / 1e9:.2f} G")

    # ---------------------------------------------------
    # Method 2: fvcore
    # ---------------------------------------------------
    flops_analyzer = FlopCountAnalysis(model, inputs)
    
    # fvcore는 기본적으로 MACs를 계산합니다.
    # total()을 호출하여 전체 연산량을 계산
    flops_fvcore_raw = flops_analyzer.total()
    
    # 지원하지 않는 연산자 확인 (Warning 출력)
    unsupported = flops_analyzer.unsupported_ops()
    if unsupported:
        print(f" * Warning: fvcore found unsupported ops: {unsupported}")

    # fvcore 결과 변환 (MACs -> FLOPs)
    final_flops_fvcore = flops_fvcore_raw * 2
    
    print(f"[fvcore]")
    print(f" - GFLOPs    : {final_flops_fvcore / 1e9:.2f} G")
    
    # ---------------------------------------------------
    # 비교 결론
    # ---------------------------------------------------
    diff = abs(flops_thop - final_flops_fvcore)
    print(f"\n[Conclusion]")
    print(f"Difference: {diff / 1e9:.4f} G")
    
    # 오차율 계산 및 신뢰도 판단
    if flops_thop > 0:
        error_rate = diff / flops_thop
        if error_rate < 0.05: # 5% 미만 차이
            print(">> 두 도구의 결과가 거의 일치합니다. (thop 신뢰 가능)")
        else:
            print(f">> 차이가 있습니다 (오차율 {error_rate*100:.1f}%).")
            print("   fvcore의 unsupported ops 혹은 thop의 커스텀 룰 부재를 확인해보세요.")

if __name__ == "__main__":
    compare_flops_measurement()
