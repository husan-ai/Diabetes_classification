# Diabetes Prediction using KNN Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Dataset](https://img.shields.io/badge/Dataset-Pima%20Indians-lightgrey)

## Project Overview

This project predicts whether a patient has diabetes using the **K-Nearest Neighbors (KNN)** classification algorithm. The dataset is sourced from the **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)** and contains diagnostic measurements from female patients aged 21 and above.

---

## Dataset

- **Source:** [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/diabetes.csv)
- **Samples:** 768 rows × 9 columns
- **Target:** `Outcome` → 0 = No Diabetes, 1 = Diabetes

| Feature | Description |
|--------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes hereditary function |
| `Age` | Age in years |
| `Outcome` | Class label (0 or 1) |

---

## Data Preprocessing

- Detected invalid `0` values in: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
- Replaced zeros with **median** values for each column
- Verified no missing values after imputation
- Applied **StandardScaler** for feature normalization

---

## Exploratory Data Analysis

- **Correlation Heatmap** to analyze feature relationships
- Key finding: `Glucose` and `BMI` are most strongly correlated with `Outcome`

---

## Model: K-Nearest Neighbors (KNN)

- Train/Test split: **90% / 10%** (`random_state=42`)
- Used **5-Fold Cross-Validation** with Jaccard scoring to find the optimal `k`
- Tested `k` values from 1 to 30
- **Optimal K found: 29** (via cross-validation)
- Final model trained with `n_neighbors=23`

---

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 74.0% |
| **F1 Score** | 0.60 |
| **Recall** | 0.556 |
| **Jaccard Index** | 0.429 |

**Confusion Matrix:**

|  | Predicted: 0 | Predicted: 1 |
|--|--|--|
| **Actual: 0** | 42 (TN) | 8 (FP) |
| **Actual: 1** | 12 (FN) | 15 (TP) |

### Analysis

- Accuracy of **74%** is decent for this dataset size
- **Recall = 0.556** indicates the model misses ~44% of actual diabetes cases (high FN count)
- Root cause: large `k` value causes the model to over-average, weakening minority class detection
- For medical diagnosis, **Recall should be higher** to minimize false negatives

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core language |
| `Pandas / NumPy` | Data manipulation |
| `Matplotlib / Seaborn` | Visualization |
| `Scikit-learn` | ML model & evaluation |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/husan-ai/diabetes-classification-knn.git
cd diabetes-classification-knn

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Diabetni_aniqlash_ML_Classification_KNN_.ipynb
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Future Improvements

- [ ] Tune `k` more carefully using weighted KNN or grid search
- [ ] Try other models: Logistic Regression, Random Forest, XGBoost
- [ ] Address class imbalance using SMOTE or class weighting
- [ ] Improve **Recall** to reduce false negatives (critical for medical diagnosis)
- [ ] Deploy as a web app using **Flask** or **Streamlit**

---

## Author

**Husan**  
ML | DL | NLP  
GitHub: [@husan-ai](https://github.com/husan-ai)
