import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 설정 및 캡처 (이 부분은 이전과 동일합니다) ---

def capture_and_save_color_frame(filename="captured_image.png"):
    """
    RealSense 카메라를 초기화하고 컬러 프레임 하나를 캡처하여 파일로 저장합니다.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("RealSense 스트리밍 시작...")
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"카메라 연결 오류: {e}")
        return False
        
    for i in range(30): # 안정화
        pipeline.wait_for_frames()

    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("컬러 프레임을 가져올 수 없습니다.")
            return False

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(filename, color_image)
        print(f"컬러 이미지가 '{filename}'으로 저장되었습니다.")
        return True

    except Exception as e:
        print(f"프레임 캡처 중 오류 발생: {e}")
        return False
        
    finally:
        pipeline.stop()
        print("RealSense 스트리밍 중지.")


# --- 이미지 처리 (임의 각도 회전) ---

def rotate_image(image, angle):
    """
    OpenCV를 사용하여 이미지를 원하는 각도(angle)로 회전합니다.
    angle: 시계 반대 방향(counter-clockwise) 각도입니다.
           (예: 시계 방향 90도 회전하려면 angle=-90을 사용)
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

# --- (이 함수는 현재 사용되지 않지만 정의는 그대로 둡니다) ---
def crop_image(image, x, y, width, height):
    """
    NumPy 배열 슬라이싱을 사용하여 이미지를 자릅니다.
    (x, y): 좌상단 모서리 좌표
    (width, height): 자를 영역의 너비와 높이
    """
    cropped = image[y:y + height, x:x + width]
    return cropped

# --- 메인 실행 ---

if __name__ == "__main__":
    
    # 1. 이미지 캡처 및 저장
    original_filename = "realsense_color_original.png"
    if not capture_and_save_color_frame(original_filename):
        print("이미지 캡처에 실패했습니다. 프로그램을 종료합니다.")
        exit()

    # 2. 저장된 이미지 로드
    original_image = cv2.imread(original_filename)
    if original_image is None:
        print(f"'{original_filename}' 파일을 읽을 수 없습니다.")
        exit()

    print(f"\n원본 이미지 크기: {original_image.shape}")

    # 3. 🎯 회전 파라미터 설정 및 적용
    rotation_angle = -1.0 # 예: 시계 방향 30도 회전
    rotated_image = rotate_image(original_image, rotation_angle)
    cv2.imwrite("realsense_color_rotated.png", rotated_image)
    print(f"이미지를 {rotation_angle}도로 회전하여 'realsense_color_rotated.png'로 저장했습니다.")
    print(f"회전된 이미지 크기: {rotated_image.shape}")


    # 4. 🎯 자르기 영역 미리보기 파라미터 설정 및 시각화
    
    # **********************************************
    # ** 원하는 자르기 영역 파라미터를 아래에 입력하세요. **
    # ** 이 값들을 기반으로 직사각형이 그려집니다. **
    # **********************************************
    crop_x = 150
    crop_y = 100
    crop_width = 300
    crop_height = 200
    
    # 자르기 영역을 시각화할 이미지 복사본 생성
    image_with_crop_preview = rotated_image.copy()

    # OpenCV의 rectangle 함수를 사용하여 직사각형 그리기
    # cv2.rectangle(img, pt1, pt2, color, thickness)
    # pt1: 직사각형의 좌상단 (x, y)
    # pt2: 직사각형의 우하단 (x + width, y + height)
    # color: BGR 색상 (예: (0, 255, 0)은 녹색)
    # thickness: 선의 두께 (양수면 선, -1이면 채우기)
    
    # 자르기 영역을 나타내는 직사각형을 빨간색으로 그립니다.
    # 두께는 2픽셀
    cv2.rectangle(
        image_with_crop_preview, 
        (crop_x, crop_y), 
        (crop_x + crop_width, crop_y + crop_height), 
        (0, 0, 255), # BGR: 빨간색
        2
    )

    # 미리보기 이미지 저장
    #cv2.imwrite("realsense_color_crop_preview.png", image_with_crop_preview)
    print(f"설정된 자르기 영역을 표시한 미리보기 이미지를 'realsense_color_crop_preview.png'로 저장했습니다.")
    
    # (선택 사항) 미리보기 이미지 화면에 보여주기
    print("\n'realsense_color_crop_preview.png' 파일을 확인하여 자르기 영역을 확인하세요.")
    print("창을 닫으려면 아무 키나 누르세요.")
    
    cv2.imshow("Crop Preview on Rotated Image", image_with_crop_preview)
    cv2.waitKey(0) # 아무 키 입력 대기
    cv2.destroyAllWindows() # 모든 OpenCV 창 닫기

    # --- 5. 본격적인 자르기 (현재 주석 처리됨) ---
    # crop_x, crop_y, crop_width, crop_height 값을 확인한 후,
    # 아래 주석을 해제하고 다시 실행하면 이미지가 잘립니다.
    
    # print("\n(주석 처리된) 자르기 작업은 현재 비활성화되어 있습니다.")
    # cropped_image = crop_image(rotated_image, crop_x, crop_y, crop_width, crop_height)
    # cv2.imwrite("realsense_color_cropped.png", cropped_image)
    # print(f"이미지를 (x={crop_x}, y={crop_y}, w={crop_width}, h={crop_height})로 잘라 'realsense_color_cropped.png'로 저장했습니다.")
    # print(f"잘린 이미지 크기: {cropped_image.shape}")

    print("\n모든 작업이 완료되었습니다.")