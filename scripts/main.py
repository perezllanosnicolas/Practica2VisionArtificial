import argparse
import os
import glob
import cv2
import numpy as np
import sys

# Inyección de ruta base para resolución absoluta del paquete 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Importaciones corregidas según la estructura del árbol de directorios
from src.panel_ocr.detector_mser import PanelDetectorMSER
from src.classifiers.knn_hog_classifier import KnnHogClassifier
from src.panel_ocr.main_panels_ocr import extract_character_boxes, find_lines_ransac

# ==========================================
# SUPRESIÓN DE NO-MÁXIMOS (NMS) PARA PANELES
# ==========================================
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


def parse_args():
    parser = argparse.ArgumentParser(description="Detector MSER de Paneles de Autopista y Lector OCR")
    parser.add_argument("--train_path", type=str, required=True, help="Ruta al directorio de entrenamiento OCR")
    parser.add_argument("--test_path", type=str, required=True, help="Ruta al directorio de imágenes de test completas")
    parser.add_argument("--visualize_ocr", action="store_true", help="Activar previsualización interactiva")
    return parser.parse_args()


def load_and_train_classifier(train_path):
    """Carga y entrenamiento del clasificador OCR."""
    images_dict = {}
    for i in range(10):
        label = str(i)
        folder = os.path.join(train_path, label)
        if os.path.exists(folder):
            images_dict[label] = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob(os.path.join(folder, '*.png'))]
            
    for sub in ['may', 'min']:
        root = os.path.join(train_path, sub)
        if os.path.exists(root):
            for char_folder in os.listdir(root):
                folder = os.path.join(root, char_folder)
                images_dict[char_folder] = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob(os.path.join(folder, '*.png'))]

    model = KnnHogClassifier(ocr_char_size=(25, 25))
    model.train(images_dict)
    return model


def main():
    args = parse_args()
    
    output_dir = "resultado_imgs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Inicializando Detector MSER...")
    detector = PanelDetectorMSER()
        
    print("Entrenando clasificador OCR...")
    ocr_model = load_and_train_classifier(args.train_path)
    print("Clasificador OCR listo.")
        
    output_txt_path = "resultado.txt"
    test_images = sorted(glob.glob(os.path.join(args.test_path, "*.png")))
    
    with open(output_txt_path, 'w') as f_out: 
        for indice, img_path in enumerate(test_images):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Error al cargar la matriz: {img_name}")
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Extracción Global
            bbox_list_raw = detector.detect(img)
            bbox_list = nms_paneles(bbox_list_raw, overlap_threshold=0.3)
            
            img_result = img.copy()
            
            for bbox in bbox_list:
                x1, y1, x2, y2, score = bbox
                
                # Protección de índices matriciales
                cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                cx2, cy2 = min(img_w, int(x2)), min(img_h, int(y2))
                
                texto_ocr_final = ""
                
                if (cx2 - cx1) > 15 and (cy2 - cy1) > 15:
                    
                    # Extracción del ROI
                    panel_roi = img[cy1:cy2, cx1:cx2]
                    
                    # Inferencia OCR Topológica
                    boxes, _, _ = extract_character_boxes(panel_roi)
                    lines = find_lines_ransac(boxes, distance_threshold=(cy2 - cy1) * 0.04)
                    
                    texto_lineas = []
                    
                    for line_idx, line in enumerate(lines):
                        chars_in_line = []
                        centros_globales = []
                        
                        colors = [(0, 255, 0), (0, 165, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                        line_color = colors[line_idx % len(colors)]
                        
                        for box in line:
                            bx, by, bw, bh = box
                            
                            # Predicción Vectorial
                            char_roi = panel_roi[by:by+bh, bx:bx+bw]
                            predicted_label = ocr_model.predict(char_roi)
                            char_str = ocr_model.label2char(predicted_label)
                            chars_in_line.append(char_str)
                            
                            # Mapeo a Coordenadas Globales
                            gx = cx1 + bx
                            gy = cy1 + by
                            centros_globales.append([int(gx + bw/2), int(gy + bh/2)])
                            
                            cv2.rectangle(img_result, (gx, gy), (gx+bw, gy+bh), line_color, 1)
                            cv2.putText(img_result, char_str, (gx, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                        texto_lineas.append("".join(chars_in_line))
                        
                        # Trazado de Recta de Consenso
                        if len(centros_globales) >= 2:
                            c_array = np.array(centros_globales, dtype=np.int32)
                            vx, vy, x0, y0 = cv2.fitLine(c_array, cv2.DIST_L2, 0, 0.01, 0.01)
                            if vx[0] != 0:
                                m = vy[0] / float(vx[0])
                                c_intercept = y0[0] - m * x0[0]
                                y_start = int(m * cx1 + c_intercept)
                                y_end = int(m * cx2 + c_intercept)
                                cv2.line(img_result, (cx1, y_start), (cx2, y_end), line_color, 1)

                    texto_ocr_final = "+".join(texto_lineas)
                
                # Volcado I/O Formato Ejercicio 4
                f_out.write(f"{img_name};{int(x1)};{int(y1)};{int(x2)};{int(y2)};panel;{score:.3f};{texto_ocr_final}\n")
                
                # Marco Global Detector
                cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img_result, f"{score:.3f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imwrite(os.path.join(output_dir, img_name), img_result)
            
            if args.visualize_ocr:
                h_screen, w_screen = 900, 1600
                if img_h > h_screen or img_w > w_screen:
                    scale = min(w_screen/img_w, h_screen/img_h)
                    img_disp = cv2.resize(img_result, (int(img_w*scale), int(img_h*scale)))
                else:
                    img_disp = img_result

                cv2.imshow("Sistema Completo (Panel + OCR)", img_disp)
                print(f"[{indice + 1}/{len(test_images)}] Procesando {img_name} - Pulsa tecla para continuar...")
                if cv2.waitKey(0) == 27:
                    break
            else:
                print(f"[{indice + 1}/{len(test_images)}] Procesado: {img_name}")
            
    print(f"Proceso finalizado. Resultados guardados en '{output_dir}/' y '{output_txt_path}'.")

if __name__ == "__main__":
    main()