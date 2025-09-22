import mediapipe as mp
import cv2

# MediaPipe Pose 모델을 사용해서 프레임에서 관절을 감지
# 연속된 프레임 간의 관절 좌표 변화량을 계
class PoseAnalyzer:산

    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        # MediaPipe의 Pose 모델과 그리기 유틸리티를 초기화합니다.
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # 이전 프레임의 랜드마크를 저장할 변수를 초기화합니다.
        self.prev_landmarks = None

    def process_frame(self, frame):
        # 단일 프레임을 처리하여 Pose Landmark를 감지하기
        # MediaPipe 모델은 RGB 이미지를 입력으로 사용하므로 변환이 필요합니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 성능 향상을 위해 이미지를 쓰기 불가(not writeable)로 설정합니다.
        frame_rgb.flags.writeable = False
        
        # 포즈 감지 수행
        results = self.pose.process(frame_rgb)

        # 다시 이미지를 쓰기 가능(writeable)으로 설정합니다.
        frame_rgb.flags.writeable = True
        return results

    def calculate_deltas(self, results):
        # 현재 프레임과 이전 프레임의 랜드마크를 비교하여 좌표 변화량을 계산하기.
        deltas = {}
        # 현재 프레임에서 랜드마크가 감지되었는지 확인합니다.
        if not results.pose_landmarks:
            # 감지되지 않았다면, 이전 랜드마크 정보도 초기화합니다.
            self.prev_landmarks = None
            return None

        current_landmarks = results.pose_landmarks.landmark

        # 이전 프레임의 랜드마크 정보가 있는 경우에만 변화량을 계산합니다.
        if self.prev_landmarks:
            for i, (current_lm, prev_lm) in enumerate(zip(current_landmarks, self.prev_landmarks)):
                delta_x = current_lm.x - prev_lm.x
                delta_y = current_lm.y - prev_lm.y
                deltas[i] = {'dx': delta_x, 'dy': delta_y}

        # 현재 랜드마크를 다음 프레임에서 사용하기 위해 '이전 랜드마크'로 저장합니다.
        self.prev_landmarks = current_landmarks
        
        # deltas가 비어있으면 (첫 프레임) None을, 아니면 딕셔너리를 반환합니다.
        return deltas if deltas else None

    def draw_landmarks(self, frame, results):
        # 프레임 위에 감지된 Pose Landmark를 그리기
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

    def close(self):
        
        # MediaPipe Pose 모델 리소스를 해제합니다.
        self.pose.close()



if __name__ == '__main__':
    # 이전에 만든 video_handler.py를 임포트하여 테스트합니다.
    from video_handler import VideoHandler

    video_path = 'your_video.mp4'  # 실제 동영상 파일 경로를 입력하세요.
    
    video_handler = VideoHandler(video_path)
    pose_analyzer = PoseAnalyzer()

    while video_handler.is_opened():
        ret, frame = video_handler.get_next_frame()
        if not ret:
            break

        # 1. 프레임에서 포즈 감지
        results = pose_analyzer.process_frame(frame)

        # 2. 좌표 변화량 계산
        deltas = pose_analyzer.calculate_deltas(results)

        # 3. 변화량 출력 (결과가 있을 경우)
        if deltas:
            # 예: 왼쪽 어깨(인덱스 11)의 변화량만 출력
            if 11 in deltas:
                dx = deltas[11]['dx']
                dy = deltas[11]['dy']
                print(f"왼쪽 어깨(11) 변화량: dx={dx:.4f}, dy={dy:.4f}")

        # 4. 프레임에 랜드마크 그리기 (시각적 확인용)
        pose_analyzer.draw_landmarks(frame, results)

        # 5. 화면에 결과 표시
        cv2.imshow('Pose Analyzer Test', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # 6. 자원 해제
    video_handler.release()
    pose_analyzer.close()
