"""
Main entry point for the OCR evaluation pipeline.

Iterates through test panel images, runs the text extraction and grouping algorithms,
evaluates character patches using the trained ML classifier, and provides an 
interactive visual debugger.
"""

import argparse
import cv2
import os
import glob
import sys
import numpy as np
from typing import List, Tuple

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.classifiers.knn_hog_classifier import KnnHogClassifier
from src.utils.dataset import load_images_dict
from src.panel_ocr.text_extractor import TextLineExtractor


def draw_visualizations(img: np.ndarray, 
                        all_boxes: List[Tuple[int, int, int, int]], 
                        valid_boxes: List[Tuple[int, int, int, int]], 
                        lines: List[List[Tuple[int, int, int, int]]], 
                        thresh: np.ndarray) -> None:
    """Renders debug windows to visualize the OCR segmentation pipeline."""
    img_w = img.shape[1]
    
    cv2.imshow("1. Binary Matrix", thresh)

    # 1. Raw Contours
    img_raw = img.copy()
    for x, y, w, h in all_boxes:
        cv2.rectangle(img_raw, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.imshow("2. Raw Contours", img_raw)

    # 2. Filtered Contours
    img_filtered = img.copy()
    for x, y, w, h in valid_boxes:
        cv2.rectangle(img_filtered, (x, y), (x+w, y+h), (255, 255, 0), 1)
    cv2.imshow("3. Valid Contours", img_filtered)

    # 3. RANSAC Topology
    img_ransac = img.copy()
    for line in lines:
        centers = []
        for x, y, w, h in line:
            cv2.rectangle(img_ransac, (x, y), (x+w, y+h), (0, 255, 0), 2)
            centers.append([int(x + w/2), int(y + h/2)])
        
        # Draw the fitted line across the character centers
        if len(centers) >= 2:
            centers_array = np.array(centers, dtype=np.int32)
            vx, vy, x0, y0 = cv2.fitLine(centers_array, cv2.DIST_L2, 0, 0.01, 0.01)
            if vx[0] != 0:
                m = vy[0] / float(vx[0])
                c = y0[0] - m * x0[0]
                cv2.line(img_ransac, (0, int(c)), (img_w, int(m * img_w + c)), (255, 255, 0), 1)

    cv2.imshow("4. RANSAC Topology", img_ransac)


def main() -> None:
    parser = argparse.ArgumentParser(description='Executes OCR classification on cropped panel images.')
    parser.add_argument('--test_path', default="data/test_ocr_panels", help='Directory containing test cropped panels')
    parser.add_argument('--train_path', default="data/train_ocr", help='Directory containing OCR training data')
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.test_path, "*.png")))
    
    if not image_paths:
        print(f"[ERROR] Directory not found or empty: {args.test_path}")
        return

    print("[INFO] Loading OCR training dataset...")
    train_dict = load_images_dict(args.train_path)

    print("[INFO] Training KnnHogClassifier...")
    model = KnnHogClassifier(ocr_char_size=(25, 25))
    model.train(train_dict)
    print("[SUCCESS] Model training complete.")

    # Initialize Extractor
    extractor = TextLineExtractor(distance_threshold_ratio=0.04)

    memory_results = {}
    current_idx = 0
    total_images = len(image_paths)

    print("\n[INTERACTIVE MODE] Use Left/Right arrows to navigate. Press ESC to exit.\n")

    while current_idx < total_images:
        img_path = image_paths[current_idx]
        img = cv2.imread(img_path)
        
        if img is None:
            current_idx += 1
            continue
            
        img_h, img_w = img.shape[:2]
        filename = os.path.basename(img_path)

        # 1. Extract and group characters
        valid_boxes, all_boxes, thresh = extractor.extract_character_boxes(img)
        lines = extractor.find_lines_ransac(valid_boxes, img_h)

        # 2. OCR Inference
        text_lines = []
        for line in lines:
            chars = []
            for x, y, w, h in line:
                roi = img[y:y+h, x:x+w]
                predicted_label = model.predict(roi)
                char_str = model.label2char(predicted_label)
                chars.append(char_str)
            text_lines.append("".join(chars))

        # Join multiple lines with a '+' separator
        final_ocr_text = "+".join(text_lines)
        
        # Format: <filename>;0;0;<w>;<h>;panel;1.0;<text>
        result_line = f"{filename};0;0;{img_w};{img_h};panel;1.0;{final_ocr_text}\n"
        memory_results[filename] = result_line
        
        print(f"[{current_idx + 1}/{total_images}] {filename} -> Detected Text: '{final_ocr_text}'")

        draw_visualizations(img, all_boxes, valid_boxes, lines, thresh) 
        
        # Interactive UI controls
        key = cv2.waitKeyEx(0)
        if key == 27: # ESC
            break
        elif key in [2424832, 65361, 81, ord('a'), ord('A')]: # Left Arrow / 'A'
            current_idx = max(0, current_idx - 1)
        else: # Right Arrow / Any other key
            current_idx += 1

    cv2.destroyAllWindows()

    with open("resultado.txt", "w") as out_file:
        out_file.writelines(memory_results.values())
            
    print("\n[SUCCESS] Execution finished. Output saved to 'resultado.txt'.")

if __name__ == "__main__":
    main()