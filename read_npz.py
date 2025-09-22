import numpy as np

# 'stats.npz' 파일이 있는 경로를 확인하세요.
try:
    with np.load('stats.npz') as data:
        mean_values = data['mean']
        std_values = data['std']

        print("--- stats.npz 파일 내용 ---")
        print("\n[평균(mean) 값]:")
        # 보기 편하게 소수점 6자리까지 출력
        np.set_printoptions(precision=6, suppress=True)
        print(mean_values.flatten()) # 1차원으로 펴서 출력

        print("\n[표준편차(std) 값]:")
        print(std_values.flatten()) # 1차원으로 펴서 출력

except FileNotFoundError:
    print("오류: 'stats.npz' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
