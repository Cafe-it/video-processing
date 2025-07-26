import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

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
    rects = []

    for contour in contours:
        # 윤곽선 근사화
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

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

    return rects

def get_seat_coordinates(image_path, debug=False):
    """좌석 사각형들의 좌표를 반환하는 메인 함수"""
    try:
        rectangles = extract_seat_boxes(image_path, debug)

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


def yolo_detect_people(image, yolo_model):
    roi_array = np.array(image)
    results = yolo_model(roi_array, verbose=False)

    people_boxes = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                if class_id == 0 and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    people_boxes.append((x1, y1, x2 - x1, y2 - y1))
    return people_boxes

def compute_overlap_ratio(boxA, boxB):
    # box = (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxB_area = boxB[2] * boxB[3]

    if boxB_area == 0:
        return 0.0

    return inter_area / boxB_area  # 초록 사각형 대비 겹친 비율


# 예시: 파란 사각형들 (rects), 초록 사각형들 (people_boxes)
def count_people_in_rects(rects, people_boxes, threshold=0.5):
    count = 0
    for rect in rects:
        for person_box in people_boxes:
            overlap = compute_overlap_ratio(rect, person_box)
            if overlap > threshold:
                count += 1
                break  # 하나만 겹치면 그 파란 사각형은 셈에 포함되므로 break
    return count

def image_processing(rects, image_path):
    model = YOLO("yolov8n.pt")
    pil_image = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_image).copy()

    boxes = yolo_detect_people(pil_image, model)

    return count_people_in_rects(rects, boxes, threshold=0.5)

# 사용 예제
if __name__ == "__main__":
    #image_path = "/Users/icecoff22/katec/video-processing/katec/IMG_0107.jpg"

    if len(sys.argv) != 3:
        print("error: 인자 부족", file=sys.stderr)
        sys.exit(1)

    try:
        image_path = sys.argv[1]
        input_seats = int(sys.argv[2])

        coordinates = get_seat_coordinates(image_path, debug=False)

        if len(coordinates) != input_seats:
            print("no")  # stdout으로 보냄
            sys.exit(0)
        
        current_seats = image_processing(coordinates, image_path)
        print(current_seats)

    except Exception as e:
        print(f"error: {str(e)}", file=sys.stderr)
        sys.exit(3)