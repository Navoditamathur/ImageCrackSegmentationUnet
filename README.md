# **Image Crack Segmentation with Wavelet U-Net**

This repository contains the implementation of a **U-Net model with Dense Cross-Level Connections and Self-Attention Mechanisms** for road crack segmentation. The model captures both fine details and contextual information to effectively identify road cracks in images. The pipeline includes training, validation, and testing with about 10,000 images, incorporating 5-fold cross-validation and early stopping for robust performance evaluation.

---

## **Features**
- **Model Architecture**:  
  A U-Net model enhanced with dense cross-level connections and self-attention mechanisms for improved segmentation accuracy.
  
- **Dataset**:  
  Trained and evaluated on a dataset of approximately 10,000 road crack images.

- **Evaluation Metrics**:  
  Performance assessed using:
  - Dice Coefficient
  - Intersection Over Union (IOU)
  - Area Under Curve (AUC)

- **Validation Strategy**:  
  - 5-Fold Cross-Validation for robustness.
  - Early stopping to prevent overfitting and ensure optimal performance.

---

## **Project Structure**
- `notebook.ipynb`: Contains the full implementation, including model creation, training, validation, and testing.
