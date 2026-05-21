# Archivo: scripts/main.py
import argparse
import os
import glob
import cv2
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.dataset import load_images_dict
from src.panel_ocr.detector_mser import PanelDetectorMSER
from src.classifiers.knn_hog_classifier import KnnHogClassifier
from src.panel_ocr.text_extractor import TextLineExtractor

def nms_paneles(boxes, overlap_threshold=0.3):
    """Filtra detecciones múltiples del mismo cartel generadas por MSER."""
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
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_threshold)[0] + 1)))
        
    return boxes_array[pick].tolist()


def main():
    parser = argparse.ArgumentParser(description="Detector MSER y Lector OCR")
    parser.add_argument("--train_path", type=str, required=True, help="Ruta entrenamiento OCR")
    parser.add_argument("--test_path", type=str, required=True, help="Ruta test completas")
    parser.add_argument("--visualize_ocr", action="store_true", help="Activar previsualización")
    args = parser.parse_args()

    detector = PanelDetectorMSER()
    extractor = TextLineExtractor()
    ocr_model = KnnHogClassifier(ocr_char_size=(25, 25))
    
    print("Entrenando clasificador OCR...")
    train_dict = load_images_dict(args.train_path)
    ocr_model.train(train_dict)
    
    test_images = sorted(glob.glob(os.path.join(args.test_path, "*.png")))
    
    with open("resultado.txt", 'w') as f_out: 
        for indice, img_path in enumerate(test_images):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_h, img_w = img.shape[:2]
            bbox_list = nms_paneles(detector.detect(img), overlap_threshold=0.3)
            img_result = img.copy()
            
            for bbox in bbox_list:
                x1, y1, x2, y2, score = [int(v) if i < 4 else v for i, v in enumerate(bbox)]
                
                # Protección de límites matriciales
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
                texto_ocr_final = ""
                
                if (x2 - x1) > 15 and (y2 - y1) > 15:
                    panel_roi = img[y1:y2, x1:x2]
                    
                    boxes, _, _ = extractor.extract_character_boxes(panel_roi)
                    lines = extractor.find_lines_ransac(boxes, panel_roi.shape[0])
                    
                    texto_lineas = []
                    for line in lines:
                        chars_in_line = []
                        for bx, by, bw, bh in line:
                            char_roi = panel_roi[by:by+bh, bx:bx+bw]
                            label = ocr_model.predict(char_roi)
                            chars_in_line.append(ocr_model.label2char(label))
                            
                            cv2.rectangle(img_result, (x1+bx, y1+by), (x1+bx+bw, y1+by+bh), (0, 255, 0), 1)
                            
                        texto_lineas.append("".join(chars_in_line))
                    
                    texto_ocr_final = "+".join(texto_lineas) 
                
                f_out.write(f"{img_name};{x1};{y1};{x2};{y2};panel;{score:.3f};{texto_ocr_final}\n")
                cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            if args.visualize_ocr:
                cv2.imshow("Sistema Completo", img_result)
                key = cv2.waitKey(0) & 0xFF 
                if key == 27 or key == ord('q'): # ESC o Q
                    break
                    
    print("Proceso finalizado. Resultados en 'resultado.txt'.")

if __name__ == "__main__":
    main()