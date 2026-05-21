import os
import glob
import cv2

def load_images_dict(base_path):
    """Carga las imágenes de entrenamiento en un diccionario organizado por clase."""
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
                        
    # Cargar dígitos numéricos
    for i in range(10):
        label = str(i)
        add_images_from_folder(os.path.join(base_path, label), label)
        
    # Cargar mayúsculas y minúsculas
    for sub_folder in ['may', 'min']:
        root = os.path.join(base_path, sub_folder)
        if os.path.exists(root):
            for char_folder in os.listdir(root):
                add_images_from_folder(os.path.join(root, char_folder), char_folder)
                
    return images_dict