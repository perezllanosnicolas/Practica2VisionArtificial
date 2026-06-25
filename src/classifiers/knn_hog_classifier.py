"""
KNN + HOG OCR Classifier.

Extracts Histogram of Oriented Gradients (HOG) features from character images
and classifies them using a K-Nearest Neighbors (KNN) algorithm.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

from src.classifiers.ocr_classifier import OCRClassifier


class KnnHogClassifier(OCRClassifier):
    """
    Optical Character Recognition using HOG descriptors coupled with
    an OpenCV K-Nearest Neighbors (KNN) classifier.
    """

    def __init__(self,
                 ocr_char_size: Tuple[int, int] = (25, 25),
                 k_neighbors: int = 3,
                 hog_win_size: Tuple[int, int] = (20, 20),
                 hog_block_size: Tuple[int, int] = (10, 10),
                 hog_block_stride: Tuple[int, int] = (5, 5),
                 hog_cell_size: Tuple[int, int] = (5, 5),
                 hog_nbins: int = 9) -> None:
        """
        Initializes the KNN classifier and configures the HOG descriptor window.
        All structural hyperparameters are exposed for easy grid-search tuning.
        """
        super().__init__(ocr_char_size)
        self.classifier_name = "KNN_HOG"

        # --- Hyperparameters ---
        self.k_neighbors = k_neighbors
        self.hog_win_size = hog_win_size

        # --- Model Initialization ---
        self.knn = cv2.ml.KNearest_create()

        self.hog = cv2.HOGDescriptor(
            _winSize=self.hog_win_size,
            _blockSize=hog_block_size,
            _blockStride=hog_block_stride,
            _cellSize=hog_cell_size,
            _nbins=hog_nbins
        )

    def extract_hog_features(self, img: np.ndarray) -> np.ndarray:
        """
        Overrides/extends base feature extraction by appending HOG computation.
        """
        # 1. Use base class to get the standardized cropped and thresholded image
        base_features = super().extract_features(img)
        
        # 2. Reshape back to image format to apply HOG
        img_processed = base_features.reshape(self.ocr_char_size).astype(np.uint8)

        # 3. Resize to the specific HOG window size
        img_hog_ready = cv2.resize(img_processed, self.hog_win_size)

        # 4. Compute HOG descriptors
        features = self.hog.compute(img_hog_ready)
        
        return features.flatten().astype(np.float32)

    def train(self, images_dict: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts HOG features and trains the K-Nearest Neighbors classifier.
        """
        x_data = []
        y_data = []

        for char_key, imgs in images_dict.items():
            label_idx = self.char2label(char_key)
            for img in imgs:
                x_data.append(self.extract_hog_features(img))
                y_data.append(label_idx)

        # KNN in OpenCV expects np.float32 for both features and labels
        x_array = np.array(x_data, dtype=np.float32)
        y_array = np.array(y_data, dtype=np.float32) 

        self.knn.train(x_array, cv2.ml.ROW_SAMPLE, y_array)
        return x_array, y_array

    def predict(self, img: np.ndarray) -> int:
        """
        Predicts the character label using the trained KNN.
        """
        feature_vector = self.extract_hog_features(img)
        feature_matrix = feature_vector.reshape(1, -1)

        # OpenCV KNN prediction returns (ret, results, neighbours, dist)
        _, results, _, _ = self.knn.findNearest(feature_matrix, k=self.k_neighbors)
        
        return int(results[0, 0])