"""
PCA + SVM OCR Classifier.

Reduces feature dimensionality using Principal Component Analysis (PCA) 
and classifies the resulting vectors using a Linear Support Vector Machine (SVM).
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple

from src.classifiers.ocr_classifier import OCRClassifier


class PcaSvmClassifier(OCRClassifier):
    """
    Optical Character Recognition using PCA dimensionality reduction 
    coupled with an OpenCV Linear SVM.
    """

    def __init__(self, ocr_char_size: Tuple[int, int] = (25, 25), 
                 variance_retention: float = 0.95, 
                 svm_c: float = 1.0) -> None:
        """
        Args:
            ocr_char_size: Target size for character normalization.
            variance_retention: Ratio of variance to retain during PCA (e.g., 0.95).
            svm_c: Penalty parameter C of the error term for the SVM.
        """
        super().__init__(ocr_char_size)
        self.classifier_name = "PCA_SVM"
        
        # --- Hyperparameters ---
        self.variance_retention = variance_retention
        self.svm_c = svm_c
        
        # --- Model Initialization ---
        self.pca = PCA(n_components=self.variance_retention)
        
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setC(self.svm_c)

    def train(self, images_dict: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features, applies PCA reduction, and trains the linear SVM.
        """
        features = []
        labels = []
        
        for char_key, imgs in images_dict.items():
            label_idx = self.char2label(char_key)
            for img in imgs:
                features.append(self.extract_features(img))
                labels.append(label_idx)
        
        x_data = np.array(features, dtype=np.float32)
        y_data = np.array(labels, dtype=np.int32)
        
        # Dimensionality reduction via PCA
        x_reduced = self.pca.fit_transform(x_data)
        
        # Train the Support Vector Machine
        self.svm.train(np.float32(x_reduced), cv2.ml.ROW_SAMPLE, y_data)
        
        return x_reduced, y_data

    def predict(self, img: np.ndarray) -> int:
        """
        Predicts the character label for a given image patch.
        """
        # Extract and flatten the raw feature vector
        feature_vector = self.extract_features(img)
        
        # Reshape for PCA transform (expects 2D array: [1, n_features])
        feature_matrix = feature_vector.reshape(1, -1)
        
        # Apply PCA projection
        reduced_vector = self.pca.transform(feature_matrix)
        
        # SVM Inference
        _, response = self.svm.predict(np.float32(reduced_vector))
        
        return int(response[0, 0])