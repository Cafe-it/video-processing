import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN

def extract_seat_boxes_improved(image_path, debug=False):
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
    
    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Blue Mask')
    
    # 방법 1: 기본 윤곽선 검출
    rects_contour = detect_rectangles_by_contour(combined_mask)
    
    # 방법 2: 허프 변환 기반 직선 검출
    rects_hough = detect_rectangles_by_hough_lines(combined_mask)
    
    # 방법 3: 템플릿 매칭 기반 검출
    rects_template = detect_rectangles_by_template_matching(combined_mask)
    
    # 방법 4: 거리 변환 기반 검출
    rects_distance = detect_rectangles_by_distance_transform(combined_mask)
    
    # 모든 방법의 결과 결합
    all_rects = rects_contour + rects_hough + rects_template + rects_distance
    
    # 중복 제거 및 필터링
    final_rects = filter_and_merge_rectangles(all_rects)
    
    if debug:
        plt.subplot(133)
        debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        for i, (x, y, w, h) in enumerate(final_rects):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        plt.imshow(debug_img)
        plt.title(f'Detected Rectangles ({len(final_rects)})')
        plt.show()
    
    print(f"검출된 사각형 개수: {len(final_rects)}")
    return final_rects

def detect_rectangles_by_contour(mask):
    """기본 윤곽선 기반 사각형 검출"""
    # 다양한 모폴로지 연산 시도
    kernels = [
        np.ones((1,1), np.uint8),
        np.ones((2,2), np.uint8),
        np.ones((3,3), np.uint8)
    ]
    
    all_rects = []
    
    for kernel in kernels:
        # 노이즈 제거
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 윤곽선 근사화
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:  # 4개 이상의 꼭짓점
                x, y, w, h = cv2.boundingRect(contour)
                
                # 크기 필터링
                if 15 < w < 300 and 15 < h < 300:
                    aspect_ratio = w / h
                    area = cv2.contourArea(contour)
                    
                    if 0.2 < aspect_ratio < 5.0 and area > 30:
                        all_rects.append((x, y, w, h))
    
    return all_rects

def detect_rectangles_by_hough_lines(mask):
    """허프 변환 기반 직선 검출로 사각형 찾기"""
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=8)
    
    if lines is None:
        return []
    
    # 수직선과 수평선 분리
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        if abs(angle) < 15 or abs(angle) > 165:  # 수평선
            horizontal_lines.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2))
        elif 75 < abs(angle) < 105:  # 수직선
            vertical_lines.append((min(y1, y2), max(y1, y2), (x1 + x2) // 2))
    
    # 교차점으로 사각형 형성
    rectangles = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            # 교차점 근처에서 사각형 가능성 확인
            x = v_line[2]
            y = h_line[2]
            
            # 주변 영역에서 다른 수직선, 수평선 찾기
            for h_line2 in horizontal_lines:
                if abs(h_line2[2] - y) > 20 and abs(h_line2[2] - y) < 100:
                    for v_line2 in vertical_lines:
                        if abs(v_line2[2] - x) > 20 and abs(v_line2[2] - x) < 100:
                            # 사각형 좌표 계산
                            x1, y1 = min(x, v_line2[2]), min(y, h_line2[2])
                            x2, y2 = max(x, v_line2[2]), max(y, h_line2[2])
                            w, h = x2 - x1, y2 - y1
                            
                            if 10 < w < 200 and 10 < h < 200:
                                rectangles.append((x1, y1, w, h))
    
    return rectangles

def detect_rectangles_by_template_matching(mask):
    """템플릿 매칭으로 사각형 패턴 검출"""
    rectangles = []
    
    # 다양한 크기의 사각형 템플릿 생성
    template_sizes = [(20, 20), (30, 30), (40, 40), (25, 35), (35, 25)]
    
    for w, h in template_sizes:
        # 사각형 테두리 템플릿 생성
        template = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (w-1, h-1), 255, 2)
        
        # 템플릿 매칭
        result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.3)
        
        for pt in zip(*locations[::-1]):
            rectangles.append((pt[0], pt[1], w, h))
    
    return rectangles

def detect_rectangles_by_distance_transform(mask):
    """거리 변환을 사용한 사각형 검출"""
    # 거리 변환
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 지역 최대값 찾기
    local_maxima = ndimage.maximum_filter(dist_transform, size=10) == dist_transform
    local_maxima = local_maxima & (dist_transform > 5)
    
    # 각 지역 최대값 주변에서 사각형 검출
    maxima_points = np.where(local_maxima)
    rectangles = []
    
    for i in range(len(maxima_points[0])):
        y, x = maxima_points[0][i], maxima_points[1][i]
        radius = int(dist_transform[y, x])
        
        # 주변 영역에서 윤곽선 찾기
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), radius*2, 255, -1)
        local_mask = cv2.bitwise_and(mask, roi_mask)
        
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                if 10 < rect_w < 150 and 10 < rect_h < 150:
                    rectangles.append((rect_x, rect_y, rect_w, rect_h))
    
    return rectangles

def filter_and_merge_rectangles(rectangles):
    """중복 제거 및 사각형 필터링"""
    if not rectangles:
        return []
    
    # 중복 제거를 위한 클러스터링
    centers = [(x + w//2, y + h//2) for x, y, w, h in rectangles]
    
    if len(centers) < 2:
        return rectangles
    
    # DBSCAN 클러스터링으로 가까운 사각형들 그룹화
    clustering = DBSCAN(eps=15, min_samples=1).fit(centers)
    labels = clustering.labels_
    
    # 각 클러스터에서 가장 적절한 사각형 선택
    unique_labels = set(labels)
    filtered_rects = []
    
    for label in unique_labels:
        if label == -1:  # 노이즈
            continue
            
        cluster_rects = [rectangles[i] for i in range(len(rectangles)) if labels[i] == label]
        
        if len(cluster_rects) == 1:
            filtered_rects.append(cluster_rects[0])
        else:
            # 클러스터 내에서 가장 큰 면적을 가진 사각형 선택
            best_rect = max(cluster_rects, key=lambda r: r[2] * r[3])
            filtered_rects.append(best_rect)
    
    # 최종 필터링
    final_rects = []
    for x, y, w, h in filtered_rects:
        # 크기 및 종횡비 최종 확인
        if 15 < w < 200 and 15 < h < 200:
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 3.0:
                final_rects.append((x, y, w, h))
    
    return final_rects

def get_seat_coordinates_improved(image_path, debug=False):
    """개선된 좌석 좌표 검출 함수"""
    try:
        rectangles = extract_seat_boxes_improved(image_path, debug)
        
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
    """이미지에 검출된 사각형 그리기"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    for i, (x, y, w, h) in enumerate(boxes):
        # 초록색 박스 (B, G, R)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 번호 표시
        cv2.putText(img, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

# 사용 예제
if __name__ == "__main__":
    image_path = "C:\\Users\\wjdtm\\Desktop\\katec\\katec\\grace_blue.jpg"
    result_path = "C:\\Users\\wjdtm\\Desktop\\katec\\katec\\grace_with_boxes_improved.jpg"

    # 개선된 함수 사용
    coordinates = get_seat_coordinates_improved(image_path, debug=True)
    draw_boxes_on_image(image_path, coordinates, result_path)
    print(f"결과 이미지 저장 완료: {result_path}")