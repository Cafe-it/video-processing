import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_seat_boxes(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("이미지 경로 확인")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 파란색 범위를 더 정확하게 설정
    blue_ranges = [
        ([100, 100, 100], [130, 255, 255]),  # 진한 파란색
        ([90, 80, 80], [125, 255, 255]),     # 중간 파란색
    ]
    
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in blue_ranges:
        lower_blue = np.array(lower)
        upper_blue = np.array(upper)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 노이즈 제거 (더 강하게)
    kernel = np.ones((2,2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # 에지 검출
    edges = cv2.Canny(combined_mask, 50, 150)
    
    # 허프 변환으로 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

    # 방법 1: 윤곽선 기반 사각형 검출
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('lines', len(lines))
    print('contours', len(contours))
    rects = []
    i = 0
    j = 0
    for contour in contours:
        # 윤곽선 근사화
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        print(j, ' ', contour)
        j+=1
        # 사각형 조건 확인
        if len(approx) == 4:  # 4개 이상의 꼭짓점
            x, y, w, h = cv2.boundingRect(contour)
            
            # 크기 필터링 (너무 작거나 큰 것 제외)
            if 50 < w and 50 < h:
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                
                # 종횡비와 면적 조건
                if 0.3 < aspect_ratio < 3.0 and area > 50:
                    rects.append((x, y, w, h))
                    print(i, ' : ', x, y, w, h)
                    i+=1

    print(len(rects))
    

    
    # 디버그 모드에서 결과 시각화
    if debug:
        print('debug')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 마스크
        axes[0, 1].imshow(combined_mask, cmap='gray')
        axes[0, 1].set_title('Blue Mask')
        axes[0, 1].axis('off')
        
        # 에지
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edges')
        axes[1, 0].axis('off')
        
        # 검출된 사각형
        result_img = img.copy()
        for i, (x, y, w, h) in enumerate(rects):
            print(rects[i])
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, f'{i+1}', (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 직선도 표시
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        axes[1, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Detected Rectangles ({len(rects)})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return rects

def get_seat_coordinates(image_path, debug=False):
    """좌석 사각형들의 좌표를 반환하는 메인 함수"""
    try:
        rectangles = extract_seat_boxes(image_path, debug)
        
        print(f"\n검출된 좌석 사각형 정보:")
        print(f"총 개수: {len(rectangles)}")
        print("좌표 정보 (x, y, width, height):")
        
        for i, (x, y, w, h) in enumerate(rectangles):
            print(f"  사각형 {i+1}: ({x}, {y}, {w}, {h})")
            print(f"    중심점: ({x + w//2}, {y + h//2})")
            print(f"    면적: {w * h}")
        
        return rectangles
        
    except Exception as e:
        print(f"에러 발생: {e}")
        return []

def draw_boxes_on_image(image_path, boxes, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # 초록색 박스 (B, G, R)
        cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 2)
        # 번호 표시
        cv2.putText(img, str(i+1), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

#def image_processing(coordinates):
    

# 사용 예제
if __name__ == "__main__":

    input_seats = "10" # <- 파라미터로 넘겨진 좌석 총 갯수. 인풋으로 받아야 함.
    image_path = "/Users/icecoff22/katec/video-processing/katec/IMG_0110.jpg"
    result_path = "/Users/icecoff22/katec/video-processing/katec/grace_with_boxes.jpg"

    coordinates = get_seat_coordinates(image_path, debug=True)
    
    # 좌석 갯수가 들어온 입력 갯수와 맞지 않으면.
    #if len(coordinates) is not input_seats:
        #TODO : 맞지않을 때 리턴 값 상의하기.
    #    return -1 
    
    #image_processing(coordinates)

    draw_boxes_on_image(image_path, coordinates, result_path)
    print(f"결과 이미지 저장 완료: {result_path}")