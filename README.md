# Distorted Visual Sequence Pattern Recognition

A PyTorch deep learning pipeline designed to decode complex alphanumeric sequences from heavily distorted CAPTCHA images. The architecture integrates custom OpenCV CLAHE preprocessing with a Convolutional Recurrent Neural Network (CRNN) optimized via Connectionist Temporal Classification (CTC) Loss[cite: 1].

---

## Key Features

* **Custom OpenCV Preprocessing**: Applied Contrast Limited Adaptive Histogram Equalization (CLAHE) routines to normalize visual noise and enhance text line contours[cite: 1].
* **CRNN Architecture**: Feature extraction using a deep CNN backbone combined with sequential context modeling via Bidirectional LSTM (BiLSTM) layers[cite: 1].
* **Alignment-Free Sequence Training**: End-to-end training using CTC Loss without requiring manual character-level bounding box annotations[cite: 1].
* **High-Precision Decoding**: Reached **98.1% sequence accuracy** and a **0.32% Character Error Rate (CER)** on a test set of 20,000 images[cite: 1].

---

## Results & Performance

| Metric | Score |
| :--- | :--- |
| **Sequence Accuracy** | **98.1%**[cite: 1] |
| **Character Error Rate (CER)** | **0.32%**[cite: 1] |
| **Dataset Size** | 20,000 distorted CAPTCHA images[cite: 1] |
| **Loss Function** | Connectionist Temporal Classification (CTC) Loss[cite: 1] |

---

## Tech Stack

* **Language**: Python
* **Deep Learning Framework**: PyTorch[cite: 1]
* **Computer Vision**: OpenCV[cite: 1]
* **Model Architecture**: CRNN (CNN + Bidirectional LSTM)[cite: 1]

---

## Project Architecture

1. **Preprocessing (`src/preprocessing.py`)**: OpenCV CLAHE routines enhance localized image contrast to isolate distorted character boundaries from background noise[cite: 1].
2. **Dataset & Encoding (`src/dataset.py`)**: Custom PyTorch `Dataset` pipeline that handles image transformations and maps character strings to numerical sequence vectors[cite: 1].
3. **Model Definition (`src/model.py`)**:
   * **CNN Layers**: Extracts spatial visual features from sequence images.
   * **BiLSTM Layers**: Captures sequential dependencies across character positions[cite: 1].
   * **Linear Projection**: Outputs per-frame character probability distributions.
4. **Optimization (`src/train.py`)**: Optimizes model weights end-to-end using CTC Loss[cite: 1].

---

## Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/distorted-visual-sequence-recognition.git
cd distorted-visual-sequence-recognition
