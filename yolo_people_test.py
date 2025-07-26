import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

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


if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # 또는 사용 중인 모델 경로
    image_path = "/Users/icecoff22/katec/video-processing/katec/IMG_0107.jpg"
    image = Image.open(image_path)

    boxes = yolo_detect_people(image, model)
    print(f"감지된 사람 수: {len(boxes)}")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"  사람 {i+1}: 위치 ({x}, {y}), 크기 ({w}, {h})")


    pil_image = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_image).copy()

    boxes = yolo_detect_people(pil_image, model)
    print(f"감지된 사람 수: {len(boxes)}")

    # 박스 그리기
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_np, f'Person {i+1}', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # BGR → RGB → 다시 PIL → 보기 좋게 저장/띄우기
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("Detected People", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rects = [[2870, 1026, 395, 461]]


    overlapped_count = count_people_in_rects(rects, boxes, threshold=0.5)
    print(f"사람이 감지된 파란 사각형 수 (50% 이상 겹침): {overlapped_count}")