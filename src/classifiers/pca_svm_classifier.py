import cv2
import numpy as np
from sklearn.decomposition import PCA
from .ocr_classifier import OCRClassifier

class PcaSvmClassifier(OCRClassifier):
    """
    Alternativa: PCA + Clasificador SVM
    """
    def __init__ (self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.pca = PCA(n_components=0.95) # Mantener el 95% de la varianza
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        
    def train (self, images_dict):
        x, y= [], []
        for char, imgs, in images_dict.items():
            label = self.char2label(char)
            for img in imgs:
                x.append(self.extract_features(img))
                y.append(label)
        
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        # Aplicar PCA
        x_reduced = self.pca.fit_transform(x)
        
        # Entrenar SVM
        self.svm.train(np.float32(x_reduced), cv2.ml.ROW_SAMPLE, y)
        return x_reduced, y
    
    def predict(self, img):
        features = self.extract_features(img).reshape(1, -1)
        features_reduced = self.pca.transform(features)
        _, result = self.svm.predict(np.float32(features_reduced))
        return int(result[0][0])