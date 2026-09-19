# Digit Classification with Neural Network (L2 Regularization)

A simple multi-layer neural network built with **TensorFlow/Keras** that classifies handwritten digits from the classic scikit-learn `digits` dataset.

## Features
- 3 hidden layers (64 → 32 → 16 neurons) with ReLU activation
- L2 regularization (`λ = 0.01`) on all dense layers
- Input standardization using `StandardScaler`
- Softmax output for 10-class classification
- SGD optimizer + Sparse Categorical Crossentropy loss

## Dataset
- **Source**: `sklearn.datasets.load_digits`
- **Shape**: 1797 samples × 64 features (8×8 pixel images)
- **Classes**: Digits 0–9

## Requirements
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python digit_classifier.py
```

## Model Architecture
```
Input (64,)
  ↓
Dense(64, relu) + L2(0.01)
  ↓
Dense(32, relu) + L2(0.01)
  ↓
Dense(16, relu) + L2(0.01)
  ↓
Dense(10, softmax)
```

## Training Details
- Train/Test split: 80/20
- Epochs: 30
- Optimizer: SGD
- Loss: Sparse Categorical Crossentropy

## Results
After training, the script prints a full classification report (precision, recall, F1-score) on the test set.

## Author
SHAM.A
