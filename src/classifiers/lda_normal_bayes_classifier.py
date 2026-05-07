# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminar la implementación
# Modificar como se crea conveniente (incluyendo métodos y parámetros), únicamente es una guía.

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        
        x = []
        y = []
        
        for key in images_dict:
            label = self.char2label(key)
            for img in images_dict[key]:
                features = self.extract_features(img)
                x.append(features)
                y.append(label)
                
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        self.lda = LinearDiscriminantAnalysis()
        x_reduced = self.lda.fit_transform(x, y)
        
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(np.float32(x_reduced), cv2.ml.ROW_SAMPLE, y)

        return x_reduced, y


    def predict(self, img):
        
        features = self.extract_features(img)
        features = np.array([features], dtype=np.float32)
        
        #Proyectar en el espacio LDA
        features_reduced = self.lda.transform(features)
        _, results = self.classifier.predict(np.float32(features_reduced))

        return int(results[0][0])



