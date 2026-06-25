"""
Abstract Base Class for Optical Character Recognition (OCR) classifiers.
Defines the strict interface and shared feature extraction utilities.
"""

import string
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class OCRClassifier(ABC):
    """
    Abstract base class enforcing a strict contract for all OCR models.
    Provides shared utilities for label encoding and basic feature extraction.
    """

    def __init__(self, ocr_char_size: Tuple[int, int] = (25, 25)) -> None:
        self.ocr_char_size = ocr_char_size
        self.classifier_name = "BaseOCR"
        
        # Character map: '0'-'9', 'a'-'z', 'A'-'Z'
        self._all_chars = '0123456789' + string.ascii_letters

    def char2label(self, c: str) -> int:
        """Encodes a character to an integer label (1-indexed)."""
        return self._all_chars.find(c) + 1

    def label2char(self, label: int) -> str:
        """Decodes an integer label back to a character."""
        if label < 1 or label > len(self._all_chars):
            return '?'
        return self._all_chars[label - 1]

    def get_labels_dict(self, images_dict: Dict[str, List[np.ndarray]]) -> List[int]:
        """Extracts a flat list of expected labels from an images dictionary."""
        responses = []
        for char_key, imgs in images_dict.items():
            label = self.char2label(char_key)
            responses.extend([label] * len(imgs))
        return responses

    def predict_dict(self, images_dict: Dict[str, List[np.ndarray]]) -> List[int]:
        """Runs batch prediction over an entire dictionary of images."""
        responses = []
        for _, imgs in images_dict.items():
            for img in imgs:
                responses.append(self.predict(img))
        return responses

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Shared preprocessing pipeline: Adaptive thresholding, bounding box 
        extraction (preserving detached components like 'i' or 'j' dots), 
        resizing, and flattening.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        
        # Local adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Custom heuristic: Encompass ALL contours to prevent amputating floating dots ('i', 'j')
            x_min = min([cv2.boundingRect(c)[0] for c in contours])
            y_min = min([cv2.boundingRect(c)[1] for c in contours])
            
            x_max_list = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours]
            y_max_list = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours]
            x_max = max(x_max_list)
            y_max = max(y_max_list)
            
            # Crop tight bounding box
            cropped = thresh[y_min:y_max, x_min:x_max]
        else:
            # Fallback if no contours found
            cropped = thresh
            
        # Standardize size for the ML model
        resized = cv2.resize(cropped, self.ocr_char_size, interpolation=cv2.INTER_AREA)
        
        # Return flattened vector
        return resized.reshape(-1).astype(np.float32)

    @abstractmethod
    def train(self, images_dict: Dict[str, List[np.ndarray]]) -> Any:
        """Trains the underlying machine learning model."""
        pass

    @abstractmethod
    def predict(self, img: np.ndarray) -> int:
        """Predicts the label for a single character image."""
        pass