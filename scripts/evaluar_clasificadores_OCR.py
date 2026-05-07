# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import matplotlib.pyplot as plt
import cv2
import string
import numpy as np
import sklearn
import sklearn.metrics

from src.classifiers.knn_hog_classifier import KnnHogClassifier
from src.classifiers.lda_normal_bayes_classifier import LdaNormalBayesClassifier
from src.classifiers.pca_svm_classifier import PcaSvmClassifier


def load_images_dict(base_path):
    
    images_dict = {}
    def add_images_from_folder(folder_path, label):
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, '*.png'))
            if files:
                if label not in images_dict:
                    images_dict[label] = []
                for f in files:
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images_dict[label].append(img)
    
    for i in range(10):
        label = str(i)
        add_images_from_folder(os.path.join(base_path, label), label)
        
    may_root = os.path.join(base_path, 'may')
    if os.path.exists(may_root):
        for char_folder in os.listdir(may_root):
            add_images_from_folder(os.path.join(may_root, char_folder), char_folder)
            
    min_root = os.path.join(base_path, 'min')
    if os.path.exists(min_root):
        for char_folder in os.listdir(min_root):
            add_images_from_folder(os.path.join(min_root, char_folder), char_folder)
    return images_dict

def plot_confusion_matrix(cm,class_names, title='Confusion matrix', cmap='Blues'):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.figure(figsize=(10, 10))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24, pad=30)
    tick_marks = np.arange(len(class_names))
    
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            valor = cm[y, x]
            if valor > 0:
                ax.annotate(str(valor), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=8,
                            # Si la celda es muy oscura (valores altos), ponemos el texto en blanco
                            color="white" if valor > (cm.max() / 2) else "black")
                
    plt.tight_layout()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        '--classifier', type=str,choices=['lda', 'knn', 'svm'], default="lda", help='Classifier string name')
    parser.add_argument(
        '--train_path', default="./data/train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./data/test_ocr", help='Select the validation data dir')
    args = parser.parse_args()

    print("Cargando datos de entrenamiento...")
    train_dict = load_images_dict(args.train_path)
    
    print("Cargando datos de validación...")
    val_dict = load_images_dict(args.validation_path)
    
    #Instanciar clasificador elegido
    
    if args.classifier == 'lda':
        model = LdaNormalBayesClassifier(ocr_char_size=(25,25))
    elif args.classifier == 'knn':
        model = KnnHogClassifier(ocr_char_size=(25,25))
    elif args.classifier == 'svm':
        model = PcaSvmClassifier(ocr_char_size=(25,25))
    print(f"Entrenado modelo {args.classifier} ...")
    model.train(train_dict)
    
    print("Evaluando sobre el conjunto de validación...")
    gt_labels= model.get_labels_dict(val_dict)
    predicted_labels = model.predict_dict(val_dict)
    
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    print(f"Accuracy = {accuracy * 100:.2f}%")
    
    all_chars= '0123456789' + string.ascii_letters
    class_names = list(all_chars)
    labels_list = list(range(1,63))
    
    cm = sklearn.metrics.confusion_matrix(gt_labels, predicted_labels, labels=labels_list)
    plot_confusion_matrix(cm, class_names, title=f"Matriz de Confusión - {args.classifier}")
    plt.show()
    

