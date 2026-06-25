"""
LDA + Normal Bayes OCR Classifier.

Reduces feature dimensionality using Linear Discriminant Analysis (LDA)
and classifies the resulting vectors using a Normal Bayes Classifier.
"""

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Dict, List, Tuple

from src.classifiers.ocr_classifier import OCRClassifier


class LdaNormalBayesClassifier(OCRClassifier):
    """
    Optical Character Recognition using LDA dimensionality reduction
    coupled with an OpenCV Normal Bayes classifier.
    """

    def __init__(self, ocr_char_size: Tuple[int, int] = (25, 25)) -> None:
        super().__init__(ocr_char_size)
        self.classifier_name = "LDA_NORMAL_BAYES"

        # --- Model Initialization ---
        self.lda = LinearDiscriminantAnalysis()
        self.bayes_classifier = cv2.ml.NormalBayesClassifier_create()

    def train(self, images_dict: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features, applies LDA reduction, and trains the Normal Bayes model.
        """
        x_data = []
        y_data = []

        for char_key, imgs in images_dict.items():
            label_idx = self.char2label(char_key)
            for img in imgs:
                x_data.append(self.extract_features(img))
                y_data.append(label_idx)

        x_array = np.array(x_data, dtype=np.float32)
        y_array = np.array(y_data, dtype=np.int32)

        # Fit and transform using Scikit-Learn's LDA
        x_reduced = self.lda.fit_transform(x_array, y_array)

        # Train the OpenCV Normal Bayes Classifier
        self.bayes_classifier.train(np.float32(x_reduced), cv2.ml.ROW_SAMPLE, y_array)

        return x_reduced, y_array

    def predict(self, img: np.ndarray) -> int:
        """
        Predicts the character label by projecting the feature vector
        into the LDA space and evaluating it with Normal Bayes.
        """
        feature_vector = self.extract_features(img)
        feature_matrix = feature_vector.reshape(1, -1)

        # Project into the pre-trained LDA space
        reduced_matrix = self.lda.transform(feature_matrix)

        # Normal Bayes Inference
        _, responses = self.bayes_classifier.predict(np.float32(reduced_matrix))

        return int(responses[0, 0])