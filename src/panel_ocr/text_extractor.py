import cv2
import numpy as np
import math
import warnings
from sklearn.linear_model import RANSACRegressor, LinearRegression

class TextLineExtractor:
    """Clase encargada de localizar caracteres individuales y agruparlos en líneas usando RANSAC."""
    
    def __init__(self, distance_threshold_ratio=0.04):
        self.distance_threshold_ratio = distance_threshold_ratio

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        img_h, _ = gray.shape
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        block_size = max(25, int(img_h / 3))
        block_size = block_size + 1 if block_size % 2 == 0 else block_size
        
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -8)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    def extract_character_boxes(self, image):
        thresh = self.preprocess_image(image)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        img_h, img_w = image.shape[:2]
        margin_x, margin_y = int(img_w * 0.03), int(img_h * 0.03)
        valid_boxes = []
        all_boxes = []

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                all_boxes.append([x, y, w, h])
                
                if hierarchy[0][i][3] != -1: continue # Ignorar huecos internos
                if x <= margin_x or y <= margin_y or (x + w) >= (img_w - margin_x) or (y + h) >= (img_h - margin_y): continue
                
                bbox_area = w * h
                if not (20 <= bbox_area <= img_h * img_w * 0.25): continue
                
                aspect_ratio = w / float(h)
                if not (0.15 <= aspect_ratio <= 4.0): continue

                valid_boxes.append([x, y, w, h])

        # Filtrado de contenedores (cajas anidadas)
        final_boxes = []
        for i, b1 in enumerate(valid_boxes):
            is_container = False
            for j, b2 in enumerate(valid_boxes):
                if i == j: continue
                cx2, cy2 = b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0
                if b1[0] < cx2 < (b1[0] + b1[2]) and b1[1] < cy2 < (b1[1] + b1[3]):
                    is_container = True
                    break
            if not is_container:
                final_boxes.append(b1)

        return final_boxes, all_boxes, thresh

    def find_lines_ransac(self, boxes, img_h):
        remaining_boxes = list(boxes)
        lines = []
        dist_thresh = img_h * self.distance_threshold_ratio

        def is_model_valid(estimator, X, y):
            slope = estimator.coef_[0]
            return math.degrees(math.atan(abs(slope))) <= 8.0

        while len(remaining_boxes) >= 2:
            X = np.array([[b[0] + b[2] / 2.0] for b in remaining_boxes])
            y = np.array([b[1] + b[3] / 2.0 for b in remaining_boxes])

            ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=2, residual_threshold=dist_thresh,
                                     is_model_valid=is_model_valid, max_trials=500, loss='absolute_error')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ransac.fit(X, y)
            except ValueError:
                break

            inlier_mask = ransac.inlier_mask_
            current_line = [remaining_boxes[i] for i in range(len(remaining_boxes)) if inlier_mask[i]]
            
            if len(current_line) >= 2:
                current_line.sort(key=lambda b: b[0]) # Ordenar de Izquierda a Derecha
                lines.append(current_line)

            for i in sorted([idx for idx, val in enumerate(inlier_mask) if val], reverse=True):
                del remaining_boxes[i]

        lines.sort(key=lambda line: sum([b[1] + b[3]/2.0 for b in line]) / len(line)) # Arriba a abajo
        return lines