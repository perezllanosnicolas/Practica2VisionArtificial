"""
OCR Models Evaluator.

Trains and evaluates optical character recognition classifiers (KNN, SVM, LDA)
using a validation dataset. Outputs the overall accuracy and plots a 
Confusion Matrix to analyze per-class performance.
"""

import sys
import os
import argparse
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import List

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifiers.knn_hog_classifier import KnnHogClassifier
from src.classifiers.lda_normal_bayes_classifier import LdaNormalBayesClassifier
from src.classifiers.pca_svm_classifier import PcaSvmClassifier
from src.utils.dataset import load_images_dict


def plot_confusion_matrix(cm: np.ndarray, class_names: str, title: str = 'Confusion Matrix', cmap: str = 'Blues') -> None:
    """
    Renders and displays a stylized confusion matrix using Matplotlib.
    
    Args:
        cm (np.ndarray): The calculated confusion matrix.
        class_names (str): String containing all possible characters in order.
        title (str): Title of the plot.
        cmap (str): Matplotlib color map.
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20, pad=20)
    
    tick_marks = np.arange(len(class_names))
    # Convert string of characters to a list for ticks
    char_list = list(class_names)
    
    plt.xticks(tick_marks, char_list, fontsize=8)
    plt.yticks(tick_marks, char_list, fontsize=8)
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluates OCR classifiers on a validation dataset.")
    parser.add_argument('--classifier', type=str, required=True, choices=['lda', 'knn', 'svm'], 
                        help="Select the classifier to evaluate (lda, knn, svm).")
    parser.add_argument('--train_path', type=str, default="data/train_ocr", 
                        help="Path to the training dataset directory.")
    parser.add_argument('--val_path', type=str, default="data/test_ocr", 
                        help="Path to the validation dataset directory.")
    args = parser.parse_args()

    print("[INFO] Loading training data...")
    train_dict = load_images_dict(args.train_path)
    
    print("[INFO] Loading validation data...")
    val_dict = load_images_dict(args.val_path)
    
    # Instantiate the selected classifier
    if args.classifier == 'lda':
        model = LdaNormalBayesClassifier(ocr_char_size=(25, 25))
    elif args.classifier == 'knn':
        model = KnnHogClassifier(ocr_char_size=(25, 25))
    elif args.classifier == 'svm':
        model = PcaSvmClassifier(ocr_char_size=(25, 25))
        
    print(f"[INFO] Training the {model.classifier_name} model. This may take a moment...")
    model.train(train_dict)
    
    print("[INFO] Evaluating on the validation set...")
    gt_labels = model.get_labels_dict(val_dict)
    predicted_labels = model.predict_dict(val_dict)
    
    # Calculate and display metrics
    accuracy = accuracy_score(gt_labels, predicted_labels)
    print(f"\n[SUCCESS] Model Accuracy: {accuracy * 100:.2f}%\n")
    
    print("[INFO] Generating Confusion Matrix...")
    all_chars = '0123456789' + string.ascii_letters
    cm = confusion_matrix(gt_labels, predicted_labels)
    
    plot_confusion_matrix(cm, class_names=all_chars, title=f'{model.classifier_name} - Confusion Matrix')


if __name__ == "__main__":
    main()