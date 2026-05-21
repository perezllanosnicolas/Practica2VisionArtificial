# Práctica Obligatoria 2 – OCR de paneles de autopista

Asignatura: **Visión Artificial**  
Universidad Rey Juan Carlos  

Este repositorio contiene la implementación de un sistema completo de **Reconocimiento Óptico de Caracteres (OCR)** aplicado a paneles informativos de autopista, desarrollado como la **Práctica Obligatoria 2** de la asignatura.

El sistema permite:
- Entrenar y evaluar clasificadores de caracteres.
- Leer automáticamente el texto de paneles de carretera ya recortados.
- Integrar el OCR con el detector de paneles desarrollado en la Práctica 1.

---

## Estructura del repositorio

```text
vision-ocr-paneles/
│
├── data/
│   ├── train_ocr/               # Imágenes de entrenamiento de caracteres
│   ├── test_ocr/                # Imágenes de validación/test de caracteres
│   ├── test_ocr_panels/         # Paneles recortados + gt.txt
│   └── fonts/
│       └── HWYGOTH.TTF           # Fuente highway-gothic
│
├── src/
│   ├── classifiers/             # Clasificadores OCR
│   │   ├── ocr_classifier.py
│   │   └── lda_normal_bayes_classifier.py
│   │
│   ├── preprocessing/           # Segmentación y features
│   │
│   ├── panel_ocr/               # OCR sobre paneles
│   │   └── main_panels_ocr.py
│   │
│   └── utils/                   # Utilidades comunes
│
├── scripts/
│   ├── evaluar_clasificadores_OCR.py
│   ├── evaluar_resultados_test_ocr_panels.py
│   └── main.py                  # Integración con la Práctica 1
│
├── results/                     # Resultados generados
│   ├── resultado.txt
│
├── memory/
│   └── memoria_practica_2.pdf
│
├── requirements.txt
└── README.md
```

## Descripción del sistema
El sistema OCR se compone de las siguientes etapas:
1. Preprocesado y segmentación de caracteres:
-Umbralización adaptativa.
-Detección de componentes conexas.
-Filtrado geométrico y recorte.

2. Extracción de características:
-Normalización a un tamaño fijo.
-Vectorización en niveles de gris (baseline).
-Alternativas: HOG / otros descriptores.

3. Reducción de dimensionalidad:
-LDA (sistema base).
-PCA u otras alternativas (Ejercicio 2).

4. Clasificación multiclase:
-Clasificador LDA + Bayes.
-Alternativas: KNN, SVM, etc.

5. OCR sobre paneles:
-Agrupación de caracteres en líneas (RANSAC).
-Orden de lectura izquierda->derecha, arriba->abajo.
-Generación del texto final usando + como separador de líneas.

## Ejecución
Ejercicio 1 y 2 - Evaluación de clasificadores OCR:
Desde la raíz del proyecto:
```bash
python scripts/evaluar_clasificadores_OCR.py --train_path data/train_ocr --validation_path data/test_ocr --classifier lda
```
El parámetro final permite seleccionar el clasificador a evaluar. (lda, knn o svm)

Ejercicio 3 - Evaluación de OCR sobre paneles:
```bash
python src/panel_ocr/main_panels_ocr.py --train_path data/train_ocr --test_path data/test_ocr_panels
```
Nota: Este script generará un archivo resultado.txt en la raíz del proyecto.

La calidad del OCR puede evaluarse mediante:
```bash
python scripts/evaluar_resultados_test_ocr_panels.py
```

Ejercicio 4 - Sistema completo con detección de paneles:
```bash
python scripts/main.py --train_path data/train_ocr --test_path data/test_ocr_panels --visualize_ocr
```
El parámetro --visualize_ocr activa la visualización paso a paso del proceso.

## Resultados
Los resultados numéricos (accuracy, matrices de confusión, distancia de Levenshtein) se inluyen en la memoria.
El fichero resultado.txt sigue exactamente el formato exigido en el enunciado.

## Autores
- [Nicolás Pérez](https://github.com/perezllanosnicolas)
- [Rubén Pisonero](https://github.com/rpisoner)

Grado en Ingeniería de Computadores, URJC
