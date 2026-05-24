# Cancer Prediction Using Machine Learning

## Overview
This project predicts the presence of breast cancer using white blood cell data from the `wbc.csv` dataset. Two machine learning models are implemented and compared: **Logistic Regression** and **K-Nearest Neighbors (KNN)**.

## Dataset
- **File**: `wbc.csv`
- **Description**: Contains test reports on different white blood cell characteristics used to predict cancer presence
- **Target Variable**: `diagnosis` (M = Malignant, B = Benign)

## Data Processing Pipeline
1. **Clean**: Remove unnecessary columns (`id`, `Unnamed: 32`)
2. **Encode**: Convert diagnosis labels (M→1, B→0)
3. **Split**: 80% training, 20% testing
4. **Scale**: Standardize features using `StandardScaler`

## Models Evaluated

### Logistic Regression (Final Model) ⭐
| Metric | Score |
|--------|-------|
| Accuracy | 98.24% |
| F1-Score | 97.56% |
| Precision | 100% |
| Recall | 95.23% |

### K-Nearest Neighbors (KNN) with k=3
| Metric | Score |
|--------|-------|
| Accuracy | 95.61% |
| F1-Score | 93.82% |
| Precision | 97.43% |
| Recall | 90.47% |

## Results
**Logistic Regression** outperformed KNN across all evaluation metrics and was selected as the final model for breast cancer classification.

## Key Insights
- The optimal KNN parameter was determined to be k=3 based on cross-validation analysis
- Logistic Regression demonstrated excellent precision, recall, and overall classification accuracy
- Both models achieved strong performance, making them suitable for cancer prediction tasks
- Outlier detection was performed using the Interquartile Range (IQR) method

## Requirements
```
numpy
pandas
seaborn
matplotlib
scikit-learn
```

## Usage
Run the script to train both models and compare their performance:
```bash
python CancerPredictionUsingML.py
```

## Features
The model uses 30 white blood cell features including:
- Radius measurements
- Texture measurements
- Perimeter measurements
- Area measurements
- And more...

## Model Selection Rationale
While both models performed well, Logistic Regression was chosen as the final model because:
- Superior accuracy (98.24% vs 95.61%)
- Perfect precision (100%)
- Better generalization with minimal overfitting
- Simpler interpretability and faster prediction time
