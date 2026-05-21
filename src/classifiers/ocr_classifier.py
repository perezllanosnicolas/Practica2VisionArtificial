# @brief OCRClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025
#

import string
import cv2
import numpy as np


class OCRClassifier:
    """
    Classifier for Optical Character Recognition
    """

    def __init__(self, ocr_char_size=(25, 25)):
        self.ocr_char_size = ocr_char_size
        self.classifier_name = None


    def char2label(self, c):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars.find(c)+1


    def label2char(self, label):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars[label-1]

    def get_labels_dict(self, images_dict):
        responses = []
        for key in images_dict:
            for img in images_dict[key]:
                responses.append(self.char2label(key))

        return responses

    def predict_dict(self, images_dict):
        responses = []
        for key in images_dict:
            for img in images_dict[key]:
                responses.append(self.predict(img))
                
        return responses

    #Umbralización, recorte, redimensionado y vectorización de la imagen
    def extract_features(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # IDEA PROPIA: En lugar del max_contour, calculamos el Bounding Box que engloba TODOS
            # los contornos. Así evitamos amputar el punto de la 'i' o 'j'.
            x_min = min([cv2.boundingRect(c)[0] for c in contours])
            y_min = min([cv2.boundingRect(c)[1] for c in contours])
            x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
            y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])
            
            roi = gray[y_min:y_max, x_min:x_max]
        else:
            roi = gray
        
        roi_resized = cv2.resize(roi, self.ocr_char_size)
        return roi_resized.flatten() 
        
