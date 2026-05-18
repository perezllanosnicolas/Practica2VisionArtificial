import argparse
import cv2
import numpy as np
import os
import glob
import math
import warnings
from sklearn.linear_model import RANSACRegressor, LinearRegression

import glob
import sys

# Agregar el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.classifiers.knn_hog_classifier import KnnHogClassifier

# ==========================================
# 1. PREPROCESAMIENTO Y EXTRACCIÓN DE CAJAS
# ==========================================

def preprocess_image(image):
    """Convierte a escala de grises, aplica desenfoque y umbralización adaptativa."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    img_h, _ = gray.shape

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # El tamaño del bloque adaptativo depende de la altura de la imagen
    block_size = int(img_h / 3)
    block_size = block_size + 1 if block_size % 2 == 0 else block_size
    block_size = max(25, block_size)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, -8
    )

    # Operación de cierre morfológico para unir pequeños huecos en los trazos
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def filter_basic_geometry(contours, hierarchy, img_shape):
    """Filtra contornos basándose en jerarquía, márgenes, área y proporciones."""
    img_h, img_w = img_shape
    margin_x, margin_y = int(img_w * 0.03), int(img_h * 0.03)
    
    pre_valid_boxes = []
    all_boxes = []

    if hierarchy is None:
        return pre_valid_boxes, all_boxes

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        all_boxes.append([x, y, w, h])

        # Ignorar contornos internos (agujeros)
        if hierarchy[0][i][3] != -1:
            continue

        # Ignorar contornos pegados a los bordes
        if x <= margin_x or y <= margin_y or (x + w) >= (img_w - margin_x) or (y + h) >= (img_h - margin_y):
            continue

        bbox_area = w * h
        if bbox_area == 0: 
            continue
            
        aspect_ratio = w / float(h)
        
        # Filtros de área y aspecto
        if not (20 <= bbox_area <= img_h * img_w * 0.25):
            continue
        if not (0.15 <= aspect_ratio <= 4.0):
            continue

        contour_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        
        if hull_area == 0: 
            continue
        
        solidity = contour_area / float(hull_area)
        extent = contour_area / float(bbox_area)

        # Filtrar formas con baja solidez o extensión (ruido lineal o muy cóncavo)
        if solidity >= 0.25 and extent >= 0.15:
            pre_valid_boxes.append([x, y, w, h])

    return pre_valid_boxes, all_boxes

def remove_nested_boxes(boxes):
    """Elimina las cajas que contienen el centro de otras cajas (cajas contenedoras)."""
    valid_boxes = []
    for i, b1 in enumerate(boxes):
        is_container = False
        for j, b2 in enumerate(boxes):
            if i == j: 
                continue
            
            # Centro de la caja b2
            cx2 = b2[0] + b2[2] / 2.0
            cy2 = b2[1] + b2[3] / 2.0
            
            # Si el centro de b2 está dentro de b1, b1 se considera un contenedor
            if b1[0] < cx2 < (b1[0] + b1[2]) and b1[1] < cy2 < (b1[1] + b1[3]):
                is_container = True
                break
                
        if not is_container:
            valid_boxes.append(b1)
            
    return valid_boxes

def filter_by_height_variance(boxes):
    """Filtra las cajas comparando su altura con la mediana general."""
    if not boxes:
        return []

    median_h = np.median([b[3] for b in boxes])
    valid_boxes = []
    
    for box in boxes:
        w, h = box[2], box[3]
        aspect_ratio = w / float(h)
        
        # Caracteres anchos (ej. guiones): permitimos mayor tolerancia en altura
        if aspect_ratio > 2.0:
            if (0.10 * median_h) < h < (1.80 * median_h):
                valid_boxes.append(box)
        # Caracteres estándar: filtro más estricto
        else:
            if (0.35 * median_h) < h < (1.80 * median_h):
                valid_boxes.append(box)
                
    return valid_boxes

def extract_character_boxes(image):
    """Función principal (orquestadora) para obtener las cajas de los caracteres."""
    thresh = preprocess_image(image)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    img_shape = image.shape[:2]
    pre_valid_boxes, all_boxes = filter_basic_geometry(contours, hierarchy, img_shape)
    
    non_nested_boxes = remove_nested_boxes(pre_valid_boxes)
    final_boxes = filter_by_height_variance(non_nested_boxes)
            
    return final_boxes, all_boxes, thresh

# ==========================================
# 2. AGRUPACIÓN DE LÍNEAS (RANSAC)
# ==========================================

def is_valid_text_spacing(line):
    """Verifica que la separación horizontal entre caracteres sea coherente."""
    if len(line) <= 3:
        line_median_h = np.median([b[3] for b in line])
        # Calcula la mayor separación horizontal (gap) entre cajas contiguas
        max_gap = max((line[k+1][0] - (line[k][0] + line[k][2]) for k in range(len(line) - 1)), default=0)
        
        # Si la separación es desproporcionada, probablemente no sea texto
        if max_gap > (4.5 * line_median_h):
            return False
    return True

def find_lines_ransac(boxes, distance_threshold=10):
    """Agrupa cajas en líneas de texto usando RANSAC sobre su posición vertical."""
    remaining_boxes = list(boxes)
    lines = []

    def is_model_valid(estimator, X, y):
        """Valida que la línea encontrada no tenga una pendiente excesiva (máx 8 grados)."""
        slope = estimator.coef_[0]
        angle = math.degrees(math.atan(abs(slope)))
        return angle <= 8.0

    while len(remaining_boxes) >= 2:
        # Preparar coordenadas (Centro X, Centro Y)
        X = np.array([[b[0] + b[2] / 2.0] for b in remaining_boxes])
        y = np.array([b[1] + b[3] / 2.0 for b in remaining_boxes])

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=2,
            residual_threshold=distance_threshold,
            is_model_valid=is_model_valid,
            max_trials=500,
            loss='absolute_error'
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ransac.fit(X, y)
        except ValueError:
            break # RANSAC falla si no encuentra modelo válido

        inlier_mask = ransac.inlier_mask_
        current_line_boxes = [remaining_boxes[i] for i in range(len(remaining_boxes)) if inlier_mask[i]]
        
        # Filtrar cajas de la línea que se desvían demasiado de la altura mediana de esa línea
        if len(current_line_boxes) >= 2:
            h_base = np.median([b[3] for b in current_line_boxes])
            filtered_line = [b for b in current_line_boxes if 0.5 < (b[3] / h_base) < 2.0]
        else:
            filtered_line = current_line_boxes

        if len(filtered_line) >= 2:
            filtered_line.sort(key=lambda b: b[0])
            if is_valid_text_spacing(filtered_line):
                lines.append(filtered_line)

        # Eliminar las cajas procesadas de la lista general para la siguiente iteración
        for i in sorted([idx for idx, val in enumerate(inlier_mask) if val], reverse=True):
            del remaining_boxes[i]

    # Ordenar las líneas resultantes de arriba hacia abajo
    lines.sort(key=lambda line: sum([b[1] + b[3]/2.0 for b in line]) / len(line))
    return lines

# ==========================================
# 3. CLASIFICADOR Y UTILIDADES
# ==========================================

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

# ==========================================
# 4. FLUJO PRINCIPAL
# ==========================================

def load_images_dict(base_path):
    images_dict = {}
    def add_images_from_folder(folder_path, label):
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, '*.png'))
            if files:
                if label not in images_dict:
                    images_dict[label] = []
                for f in files:
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images_dict[label].append(img)
    
    for i in range(10):
        label = str(i)
        add_images_from_folder(os.path.join(base_path, label), label)
        
    may_root = os.path.join(base_path, 'may')
    if os.path.exists(may_root):
        for char_folder in os.listdir(may_root):
            add_images_from_folder(os.path.join(may_root, char_folder), char_folder)
            
    min_root = os.path.join(base_path, 'min')
    if os.path.exists(min_root):
        for char_folder in os.listdir(min_root):
            add_images_from_folder(os.path.join(min_root, char_folder), char_folder)
    return images_dict

def main():
    parser = argparse.ArgumentParser(description='Ejecuta OCR sobre un conjunto de imágenes de prueba.')
    parser.add_argument('--test_path', default="data/test_ocr_panels", help='Directorio con los recortes')
    parser.add_argument('--train_path', default="data/train_ocr", help='Directorio con datos de entrenamiento OCR')
    args = parser.parse_args()

    search_path = os.path.join(args.test_path, "*.png")
    image_paths = sorted(glob.glob(search_path))
    
    if not image_paths:
        print(f"Error: Directorio inexistente o vacío ({args.test_path})")
        return

    print("Cargando imágenes de entrenamiento...")
    train_dict = load_images_dict(args.train_path)

    print("Entrenando KnnHogClassifier...")
    model = KnnHogClassifier(ocr_char_size=(25, 25))
    model.train(train_dict)
    print("Modelo entrenado.")

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

        boxes, all_boxes, thresh = extract_character_boxes(img)
        lines = find_lines_ransac(boxes, distance_threshold=img_h * 0.04)

        # Lectura OCR real
        texto_lineas = []
        for line in lines:
            # Predecir cada recuadro (carácter) en la línea
            chars = []
            for x, y, w, h in line:
                roi = img[y:y+h, x:x+w]
                predicted_label = model.predict(roi)
                char_str = model.label2char(predicted_label)
                chars.append(char_str)
            texto_linea_actual = "".join(chars)
            texto_lineas.append(texto_linea_actual)

        texto_ocr_final = "+".join(texto_lineas)
        
        # Guardar en memoria el resultado
        linea_txt = f"{filename};0;0;{img_w};{img_h};panel;1.0;{texto_ocr_final}\n"
        resultados_memoria[filename] = linea_txt
        
        print(f"[{indice_actual + 1}/{total_imagenes}] {filename} | Flechas Izq/Der: Navegar | ESC: Salir")

        # Visualizaciones
        draw_visualizations(img, all_boxes, boxes, lines, thresh)
        
        # Control de navegación en ventanas
        key = cv2.waitKeyEx(0)
        if key == 27: # ESC
            break
        elif key in [2424832, 65361, 81, ord('a'), ord('A')]: # Flecha Izq o 'A'
            indice_actual = max(0, indice_actual - 1)
        else: # Siguiente imagen
            indice_actual += 1

    cv2.destroyAllWindows()

    # Volcar memoria a archivo
    with open("resultado.txt", "w") as out_file:
        out_file.writelines(resultados_memoria.values())
            
    print("Ejecución finalizada. Archivo 'resultado.txt' generado exitosamente.")

if __name__ == "__main__":
    main()