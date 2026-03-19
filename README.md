# Multi-Domain Fractal Tumor Diagnosis System

An unsupervised machine learning pipeline fusing multi-domain (spatial, polar, frequency) fractal dimensions to calculate quantitative anomaly scores, providing a non-invasive mathematical proxy for tumor malignancy.This repository provides an unsupervised machine learning pipeline for medical image analysis, focusing on fractal dimension extraction and anomaly detection. The system extracts the Box-Counting Dimension from 256x256 TIFF grayscale images across three distinct domains: Spatial, Polar, and Frequency. By constructing a 3D feature space, it utilizes an Isolation Forest and K-Means clustering to output an anomaly score, which serves as a quantitative proxy for tumor malignancy or disease probability.

## Key Features

- **Multi-Domain Feature Fusion**: Synchronously processes morphological and textural features across spatial, polar, and frequency domains.
- **Unsupervised Detection Paradigm**: Evaluates the degree of abnormality based on the natural distribution of fractal features, bypassing the need for prior ground-truth labels.
- **Strict Data Alignment**: Features a built-in cross-validation mechanism for filenames to ensure precise multi-domain feature alignment.
- **Fault Tolerance & Visualization**: Includes an automatic mock-data generation fallback for testing environments and automatically outputs a 3D feature space clustering scatter plot.

## Prerequisites

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Scikit-learn
- Matplotlib

The corresponding files across the three domains must share the exact same filename. The default path configurations are as follows (these can be modified in the Config class within the script):
Spatial Domain Images: /data2/cz/zbk/pic_cropcir_end/
Polar Domain Images: /data2/cz/zbk/pic_cropcir_end1/
Frequency Domain Images: /data2/cz/zbk/pic_cropcir_end2/

## Output

Upon execution, the program will generate the following in the `./output_results/` directory:

* **`fractal_feature_space.png`**: A 3D fractal feature space distribution plot (color depth represents the anomaly probability).
* **Console Output**: K-Means clustering labels and Isolation Forest anomaly probability proxy values for each sample.
<img width="3000" height="2400" alt="31ff9be7567b9d9cb9e53839235e4a08" src="https://github.com/user-attachments/assets/80ad80bb-41a7-4025-9677-2f559c10d25c" />
