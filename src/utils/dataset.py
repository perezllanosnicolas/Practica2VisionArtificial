"""
Dataset utility functions.

Responsible for loading and structuring the Optical Character Recognition (OCR) 
training dataset from the filesystem into memory.
"""

import os
import glob
import cv2
import numpy as np
from typing import Dict, List


def _add_images_from_folder(images_dict: Dict[str, List[np.ndarray]], folder_path: str, label: str) -> None:
    """
    Reads all PNG images from a specified folder and appends them to the 
    dictionary under the given label key.
    """
    if os.path.exists(folder_path):
        files = glob.glob(os.path.join(folder_path, '*.png'))
        if files:
            if label not in images_dict:
                images_dict[label] = []
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images_dict[label].append(img)


def load_images_dict(base_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Loads training images into a dictionary organized by character class.
    
    Args:
        base_path (str): The root directory containing the OCR training data.
        
    Returns:
        Dict[str, List[np.ndarray]]: A dictionary where keys are character labels
                                     and values are lists of grayscale images.
    """
    images_dict: Dict[str, List[np.ndarray]] = {}
    
    # 1. Load numeric digits (Folders '0' through '9')
    for i in range(10):
        label = str(i)
        _add_images_from_folder(images_dict, os.path.join(base_path, label), label)
        
    # 2. Load uppercase and lowercase letters (Folders 'may' and 'min')
    # Note: Directory names 'may' (mayúsculas) and 'min' (minúsculas) are 
    # matched exactly to the physical filesystem structure.
    for sub_folder in ['may', 'min']:
        root = os.path.join(base_path, sub_folder)
        if os.path.exists(root):
            for char_folder in os.listdir(root):
                # char_folder represents the actual letter (e.g., 'A', 'b')
                _add_images_from_folder(images_dict, os.path.join(root, char_folder), char_folder)
                
    return images_dict