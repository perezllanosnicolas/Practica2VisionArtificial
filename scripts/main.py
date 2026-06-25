"""
End-to-End Highway Traffic Sign Detection and OCR Pipeline.

This script runs the complete system:
1. Detects highway panels in full images using MSER.
2. Crops the detected panels.
3. Extracts individual characters using morphological operations and RANSAC.
4. Recognizes the text using a trained KNN + HOG classifier.
5. Outputs the results to a structured text file and provides optional visualization.
"""

import argparse
import os
import glob
import cv2
import numpy as np
import sys
from typing import List

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.dataset import load_images_dict
from src.panel_ocr.detector_mser import PanelDetectorMSER
from src.classifiers.knn_hog_classifier import KnnHogClassifier
from src.panel_ocr.text_extractor import TextLineExtractor

def apply_nms(boxes: List[List[float]], overlap_threshold: float = 0.3) -> List[List[float]]:
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
    
    Args:
        boxes: List of bounding boxes in format [x1, y1, x2, y2, score].
        overlap_threshold: Maximum allowed Intersection over Union (IoU) overlap.
        
    Returns:
        List of filtered bounding boxes.
    """
    if len(boxes) == 0:
        return []
        
    boxes_array = np.array(boxes)
    pick = []
    
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    scores = boxes_array[:, 4]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1] 
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[1:]]
        # Delete the current index and any index that exceeds the overlap threshold
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_threshold)[0] + 1)))
        
    return boxes_array[pick].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End MSER Panel Detector and OCR Reader")
    parser.add_argument("--train_path", type=str, required=True, help="Path to OCR training dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to full test images")
    parser.add_argument("--visualize_ocr", action="store_true", help="Enable live visualization")
    args = parser.parse_args()

    print("[INFO] Initializing E2E Pipeline components...")
    detector = PanelDetectorMSER()
    extractor = TextLineExtractor()
    ocr_model = KnnHogClassifier(ocr_char_size=(25, 25))
    
    print("[INFO] Training OCR classifier. Please wait...")
    train_dict = load_images_dict(args.train_path)
    ocr_model.train(train_dict)
    print("[SUCCESS] OCR Model trained.")
    
    test_images = sorted(glob.glob(os.path.join(args.test_path, "*.png")))
    if not test_images:
        print(f"[ERROR] No images found in {args.test_path}")
        return

    output_filename = "resultado.txt"
    
    print(f"\n[INFO] Processing {len(test_images)} images...")
    
    with open(output_filename, 'w') as f_out: 
        for idx, img_path in enumerate(test_images):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 1. Detect Panels
            raw_boxes = detector.detect(img)
            bbox_list = apply_nms(raw_boxes, overlap_threshold=0.3)
            
            img_result = img.copy()
            
            for bbox in bbox_list:
                x1, y1, x2, y2, score = [int(v) if i < 4 else float(v) for i, v in enumerate(bbox)]
                
                # Boundary protection
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                
                final_ocr_text = ""
                
                # Only process if bounding box is reasonably sized
                if (x2 - x1) > 15 and (y2 - y1) > 15:
                    panel_roi = img[y1:y2, x1:x2]
                    
                    # 2. Extract Character Boxes & Group by Lines
                    char_boxes, _, _ = extractor.extract_character_boxes(panel_roi)
                    lines = extractor.find_lines_ransac(char_boxes, panel_roi.shape[0])
                    
                    text_lines = []
                    for line in lines:
                        chars_in_line = []
                        for bx, by, bw, bh in line:
                            char_roi = panel_roi[by:by+bh, bx:bx+bw]
                            
                            # 3. Predict Character
                            label = ocr_model.predict(char_roi)
                            chars_in_line.append(ocr_model.label2char(label))
                            
                            if args.visualize_ocr:
                                cv2.rectangle(img_result, (x1+bx, y1+by), (x1+bx+bw, y1+by+bh), (0, 255, 0), 1)
                            
                        text_lines.append("".join(chars_in_line))
                    
                    final_ocr_text = "+".join(text_lines) 
                
                # Write to output file
                f_out.write(f"{img_name};{x1};{y1};{x2};{y2};panel;{score:.3f};{final_ocr_text}\n")
                
                if args.visualize_ocr:
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Log progress
            print(f"[{idx + 1}/{len(test_images)}] Processed {img_name} -> Found {len(bbox_list)} panels.")
            
            if args.visualize_ocr:
                cv2.imshow("End-to-End System", img_result)
                key = cv2.waitKey(0) & 0xFF 
                if key == 27 or key == ord('q'): # ESC or Q
                    print("\n[INFO] Visualization interrupted by user.")
                    break
                    
    print(f"\n[SUCCESS] Execution finished. Results saved in '{output_filename}'.")

if __name__ == "__main__":
    main()