# Skin Lesion Segmentation and Classification

An AI-powered deep learning project for **skin lesion segmentation and multi-class skin cancer classification** using dermoscopic images from the **HAM10000 dataset**.

This system uses image preprocessing, handcrafted feature extraction, and a **Hybrid CNN-LSTM model** to improve classification accuracy and support early skin cancer detection.

## Project Overview

Skin cancer is one of the most common diseases worldwide. Early diagnosis significantly improves treatment success. Manual diagnosis can be slow, expensive, and dependent on expert availability.

This project provides an automated solution that:

- Detects and classifies skin lesions
- Segments lesion regions from skin images
- Uses deep learning for improved accuracy
- Can be deployed as a web app using Flask

## Features

- Skin lesion image preprocessing
- Hair removal using Black Hat filter + Inpainting
- Contrast enhancement using CLAHE
- Image resizing to 224x224
- Dataset balancing using under-sampling and over-sampling
- Feature extraction:
  - Color features
  - Texture features (GLCM)
  - Shape features
- Hybrid CNN + LSTM deep learning model
- Multi-class classification (7 lesion types)
- Flask web deployment support

## Dataset

### HAM10000 Dataset

Human Against Machine with 10000 Training Images

Contains **10,015 dermoscopic images** across 7 lesion classes:

- NV – Melanocytic nevi
- MEL – Melanoma
- BKL – Benign keratosis-like lesions
- BCC – Basal cell carcinoma
- AKIEC – Actinic keratoses
- VASC – Vascular lesions
- DF – Dermatofibroma

## Folder Structure

```text
project/
│── HAM10000_metadata.csv
│── lesion_features.csv
│── skin_lesion_features.csv
│── main_ann_updated.ipynb
│── main2_cnn_qnn.ipynb
│── README.md
│── requirements.txt