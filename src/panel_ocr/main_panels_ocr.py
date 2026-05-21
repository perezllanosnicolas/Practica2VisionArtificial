import argparse
import cv2
import os
import glob
import sys
import numpy as np

# Agregar el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.classifiers.knn_hog_classifier import KnnHogClassifier
from src.utils.dataset import load_images_dict
from src.panel_ocr.text_extractor import TextLineExtractor

def draw_visualizations(img, all_boxes, boxes, lines, thresh):
    """Dibuja y muestra los resultados parciales y finales en ventanas."""
    img_w = img.shape[1]
    
    cv2.imshow("1. Matriz Binaria", thresh)

    # Contornos crudos
    img_raw = img.copy()
    for x, y, w, h in all_boxes:
        cv2.rectangle(img_raw, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.imshow("2. Contornos RAW", img_raw)

    # Contornos válidos tras filtros
    img_filtered = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img_filtered, (x, y), (x+w, y+h), (255, 255, 0), 1)
    cv2.imshow("3. Contornos Validos", img_filtered)

    # Topología RANSAC
    img_ransac = img.copy()
    for line in lines:
        centers = []
        for x, y, w, h in line:
            cv2.rectangle(img_ransac, (x, y), (x+w, y+h), (0, 255, 0), 2)
            centers.append([int(x + w/2), int(y + h/2)])
        
        # Dibujar la línea ajustada sobre las cajas
        if len(centers) >= 2:
            centers_array = np.array(centers, dtype=np.int32)
            vx, vy, x0, y0 = cv2.fitLine(centers_array, cv2.DIST_L2, 0, 0.01, 0.01)
            if vx[0] != 0:
                m = vy[0] / float(vx[0])
                c = y0[0] - m * x0[0]
                cv2.line(img_ransac, (0, int(c)), (img_w, int(m * img_w + c)), (255, 255, 0), 1)

    cv2.imshow("4. Topologia RANSAC", img_ransac)


def main():
    parser = argparse.ArgumentParser(description='Ejecuta OCR sobre un conjunto de imágenes de prueba.')
    parser.add_argument('--test_path', default="data/test_ocr_panels", help='Directorio con los recortes')
    parser.add_argument('--train_path', default="data/train_ocr", help='Directorio con datos de entrenamiento OCR')
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.test_path, "*.png")))
    
    if not image_paths:
        print(f"Error: Directorio inexistente o vacío ({args.test_path})")
        return

    print("Cargando imágenes de entrenamiento...")
    train_dict = load_images_dict(args.train_path)

    print("Entrenando KnnHogClassifier...")
    model = KnnHogClassifier(ocr_char_size=(25, 25))
    model.train(train_dict)
    print("Modelo entrenado.")

    # Instanciar el extractor
    extractor = TextLineExtractor(distance_threshold_ratio=0.04)

    resultados_memoria = {}
    indice_actual = 0
    total_imagenes = len(image_paths)

    while indice_actual < total_imagenes:
        img_path = image_paths[indice_actual]
        img = cv2.imread(img_path)
        
        if img is None:
            indice_actual += 1
            continue
            
        img_h, img_w = img.shape[:2]
        filename = os.path.basename(img_path)

        boxes, all_boxes, thresh = extractor.extract_character_boxes(img)
        lines = extractor.find_lines_ransac(boxes, img_h)

        # Lectura OCR real
        texto_lineas = []
        for line in lines:
            chars = []
            for x, y, w, h in line:
                roi = img[y:y+h, x:x+w]
                predicted_label = model.predict(roi)
                char_str = model.label2char(predicted_label)
                chars.append(char_str)
            texto_lineas.append("".join(chars))

        texto_ocr_final = "+".join(texto_lineas)
        
        linea_txt = f"{filename};0;0;{img_w};{img_h};panel;1.0;{texto_ocr_final}\n"
        resultados_memoria[filename] = linea_txt
        
        print(f"[{indice_actual + 1}/{total_imagenes}] {filename} | Flechas Izq/Der: Navegar | ESC: Salir")

        draw_visualizations(img, all_boxes, boxes, lines, thresh) 
        
        key = cv2.waitKeyEx(0)
        if key == 27: # ESC
            break
        elif key in [2424832, 65361, 81, ord('a'), ord('A')]:
            indice_actual = max(0, indice_actual - 1)
        else:
            indice_actual += 1

    cv2.destroyAllWindows()

    with open("resultado.txt", "w") as out_file:
        out_file.writelines(resultados_memoria.values())
            
    print("Ejecución finalizada. Archivo 'resultado.txt' generado exitosamente.")

if __name__ == "__main__":
    main()