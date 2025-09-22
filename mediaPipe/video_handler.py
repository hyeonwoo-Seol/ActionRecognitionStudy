import cv2
import sys

# 동영상 파일을 처리하는 클래스
class VideoHandler:

    def __init__(self, video_path):
        # 처리할 동영상 파일의 경로를 가져오고 동영상 열기
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print(f"오류: '{video_path}' 동영상을 열 수 없습니다.")
            sys.exit()
        
        print("동영상이 성공적으로 열렸습니다.")

    def get_next_frame(self):
        # 동영상에서 다음 프레임 읽어오기
        return self.cap.read()

    def is_opened(self):
        """
        동영상 파일이 성공적으로 열렸는지 확인합니다.

        Returns:
            bool: 동영상이 열려있으면 True, 아니면 False.
        """
        return self.cap.isOpened()

    def release(self):
        """
        비디오 캡처 객체를 해제하고 모든 창을 닫습니다.
        """
        print("비디오 처리를 종료하고 자원을 해제합니다.")
        self.cap.release()
        cv2.destroyAllWindows()


        
if __name__ == '__main__':
    # 동영상 파일 넣어보기
    video_path = 'your_video.mp4' 

    # 1. VideoHandler 객체 생성
    video_handler = VideoHandler(video_path)

    # 2. 동영상이 열려있는 동안 반복
    while video_handler.is_opened():
        # 3. 다음 프레임 가져오기
        ret, frame = video_handler.get_next_frame()

        # 프레임이 더 이상 없으면 반복 종료
        if not ret:
            print("동영상의 끝에 도달했습니다.")
            break

        # 4. 화면에 프레임 표시
        cv2.imshow('Video Test', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 5. 자원 해제
    video_handler.release()
