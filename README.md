# Dry Bean Classification - 5 ML Models + Streamlit App
Submission by: Sarthak Rout 
BITS ID: 2026AK05289
## a. Problem Statement
Classify dry bean grains into one of 7 varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) from 16 geometric features extracted from images. The task is a multi-class classification problem; five ML models are trained on the same data and compared on six evaluation metrics.

## b. Dataset Description
- Source: UCI Machine Learning Repository - Dry Bean Dataset (ID 602)
- Instances: 13,611 | Features: 16 (area, perimeter, major/minor axis length, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, ShapeFactor1-4) | Target: `Class` (7 classes)
- No missing values; class distribution is imbalanced (Dermason largest, Bombay smallest).
- Split: 80% train / 20% test, stratified, `random_state=42`. The test split is saved as `test_data.csv`.
- Preprocessing: StandardScaler (inside pipelines for LR, KNN, SVM; not needed for Decision Tree / Gaussian NB).

## c. GitHub Repository Link
<PASTE YOUR GITHUB REPO URL HERE>

Live app: <PASTE STREAMLIT COMMUNITY CLOUD URL HERE>

## d. Models Used

### Comparison Table

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Support Vector Machine |     0.9221 | 0.9949 |      0.9359 |   0.933  | 0.9343 | 0.9059 |
| Logistic Regression    |     0.9214 | 0.9948 |      0.9354 |   0.9321 | 0.9335 | 0.905  |
| KNN                    |     0.9166 | 0.9833 |      0.932  |   0.9271 | 0.9293 | 0.8992 |
| Decision Tree          |     0.8917 | 0.9448 |      0.9084 |   0.9088 | 0.9085 | 0.8691 |
| Naive Bayes            |     0.7639 | 0.9672 |      0.7744 |   0.7694 | 0.7677 | 0.7154 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <fill using your numbers: e.g. strong linear baseline, struggles on overlapping classes such as Sira/Dermason/Barbunya> |
| Decision Tree | <fill: captures non-linear splits, fast, but may overfit; lower AUC because of hard leaf probabilities> |
| KNN | <fill: good after scaling, sensitive to k and to feature scale; slower prediction> |
| Naive Bayes | <fill: lowest, since features are highly correlated, violating independence assumption> |
| Support Vector Machine | <fill: RBF kernel models non-linear boundaries; usually best/near-best> |

**Overall winner:** Support Vector Machine (0.9221)

## Repository Structure
```
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── label_info.json
```

## How to Run
```
pip install -r requirements.txt
python train_models.py      # trains, saves models/ and test_data.csv
streamlit run app.py
```
