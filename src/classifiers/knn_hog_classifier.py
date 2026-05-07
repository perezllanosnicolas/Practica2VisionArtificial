import cv2
import numpy as np
from .ocr_classifier import OCRClassifier

class KnnHogClassifier(OCRClassifier):
    """
    Alternativa: Descriptores HOG + Clasificador K-Nearest Neighbors (KNN)
    """
    def __init__ (self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.knn = cv2.ml.KNearest_create()
        
        self.hog_win_size = (20,20)
        self.hog = cv2.HOGDescriptor(_winSize=self.hog_win_size, _blockSize=(10,10), _blockStride=(5,5), _cellSize=(5,5), _nbins=9)
        
                                    
    
    
    def extract_features(self, img):
        
        img_processed = super().extract_features(img).reshape(self.ocr_char_size)
        
        img_hog_ready = cv2.resize(img_processed, self.hog_win_size)
        
        features = self.hog.compute(img_hog_ready)
        
        return features.flatten().astype(np.float32)
    
    
    def train(self, images_dict):
        x,y = [],[]
        for char, imgs in images_dict.items():
            label = self.char2label(char)
            for img in imgs:
                x.append(self.extract_features(img))
                y.append(label)
        
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        self.knn.train(x, cv2.ml.ROW_SAMPLE, y)
        return x,y
    
    def predict(self, img):
        features = self.extract_features(img).reshape(1, -1)
        _, result, _, _ = self.knn.findNearest(features, k=3)
        return int(result[0][0])
    