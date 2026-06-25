"""
Text Line Extractor.

Responsible for segmenting individual character bounding boxes from cropped
highway panels and grouping them into logical text lines using RANSAC.
"""

import cv2
import numpy as np
import math
import warnings
from sklearn.linear_model import RANSACRegressor, LinearRegression
from typing import List, Tuple, Any

class TextLineExtractor:
    """
    Extracts characters and groups them into text lines using 
    Adaptive Thresholding, Contour topology, and RANSAC linear regression.
    """

    def __init__(self, 
                 distance_threshold_ratio: float = 0.04,
                 margin_ratio: float = 0.03,
                 min_area: int = 20,
                 max_area_ratio: float = 0.25,
                 min_aspect: float = 0.15,
                 max_aspect: float = 4.0) -> None:
        
        # --- Grouping Parameters ---
        self.distance_threshold_ratio = distance_threshold_ratio
        
        # --- Geometric Constraints ---
        self.margin_ratio = margin_ratio
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Applies adaptive thresholding to isolate dark text on light backgrounds."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        img_h = gray.shape[0]
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Dynamic block size based on image height
        block_size = max(25, int(img_h / 3))
        block_size = block_size + 1 if block_size % 2 == 0 else block_size
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, -8
        )
        
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    def _filter_containers(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Removes bounding boxes that entirely enclose other boxes (e.g., panel borders)."""
        final_boxes = []
        for i, b1 in enumerate(boxes):
            is_container = False
            for j, b2 in enumerate(boxes):
                if i == j:
                    continue
                # Check if center of b2 is inside b1
                cx2 = b2[0] + b2[2] / 2.0
                cy2 = b2[1] + b2[3] / 2.0
                if b1[0] < cx2 < (b1[0] + b1[2]) and b1[1] < cy2 < (b1[1] + b1[3]):
                    is_container = True
                    break
            if not is_container:
                final_boxes.append(b1)
        return final_boxes

    def extract_character_boxes(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Main pipeline to segment characters from the panel.
        Returns: (final_boxes, raw_boxes, binary_threshold_image)
        """
        thresh = self._preprocess_image(image)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        img_h, img_w = image.shape[:2]
        margin_x = int(img_w * self.margin_ratio)
        margin_y = int(img_h * self.margin_ratio)
        
        valid_boxes = []
        all_boxes = []

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                all_boxes.append((x, y, w, h))
                
                # Ignore internal holes (e.g., inside 'O' or 'P')
                if hierarchy[0][i][3] != -1: 
                    continue 
                    
                # Border margin filter
                if x <= margin_x or y <= margin_y or (x + w) >= (img_w - margin_x) or (y + h) >= (img_h - margin_y): 
                    continue
                
                # Area filter
                bbox_area = w * h
                if bbox_area < self.min_area or bbox_area > (img_h * img_w * self.max_area_ratio): 
                    continue
                
                # Aspect ratio filter
                aspect_ratio = w / float(h)
                if aspect_ratio < self.min_aspect or aspect_ratio > self.max_aspect: 
                    continue

                valid_boxes.append((x, y, w, h))

        final_boxes = self._filter_containers(valid_boxes)
        return final_boxes, all_boxes, thresh

    def find_lines_ransac(self, boxes: List[Tuple[int, int, int, int]], img_h: int) -> List[List[Tuple[int, int, int, int]]]:
        """
        Uses RANSAC to group character bounding boxes into logical horizontal lines.
        """
        remaining_boxes = list(boxes)
        lines = []
        dist_thresh = img_h * self.distance_threshold_ratio

        def is_model_valid(estimator: Any, X: np.ndarray, y: np.ndarray) -> bool:
            """Validates that the detected line is roughly horizontal (max 8 degrees slope)."""
            slope = estimator.coef_[0]
            return math.degrees(math.atan(abs(slope))) <= 8.0

        while len(remaining_boxes) >= 2:
            X = np.array([[b[0] + b[2] / 2.0] for b in remaining_boxes])
            y = np.array([b[1] + b[3] / 2.0 for b in remaining_boxes])

            ransac = RANSACRegressor(
                estimator=LinearRegression(), 
                min_samples=2, 
                residual_threshold=dist_thresh,
                is_model_valid=is_model_valid, 
                max_trials=500, 
                loss='absolute_error'
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ransac.fit(X, y)
            except ValueError:
                break

            inlier_mask = ransac.inlier_mask_
            current_line = [remaining_boxes[i] for i in range(len(remaining_boxes)) if inlier_mask[i]]
            
            if len(current_line) >= 2:
                # Sort characters left-to-right
                current_line.sort(key=lambda b: b[0]) 
                lines.append(current_line)

            # Remove inliers from the remaining pool
            for i in sorted([idx for idx, val in enumerate(inlier_mask) if val], reverse=True):
                del remaining_boxes[i]

        # Sort lines top-to-bottom
        lines.sort(key=lambda line: sum([b[1] + b[3]/2.0 for b in line]) / len(line))
        return lines