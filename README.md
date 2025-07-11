# 🩺 Breast Cancer Classification with ML Models

This project explores the **Breast Cancer Wisconsin Diagnostic Dataset** using multiple supervised learning models. The goal is to predict whether a tumor is **malignant (1)** or **benign (0)** based on 30 features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

---

## 📚 Dataset Overview

- **Samples**: 569
- **Features**: 30 numerical (radius, texture, perimeter, area, etc.)
- **Target**: `0` = Benign, `1` = Malignant

Dataset source: `sklearn.datasets.load_breast_cancer()`

---

## 🧠 Machine Learning Models Used

We applied and compared the following classifiers:

- ✅ Decision Tree Classifier
- ✅ Random Forest Classifier
- ✅ XGBoost Classifier
- ✅ K-Nearest Neighbors (KNN)

We performed **hyperparameter tuning** using `GridSearchCV` and evaluated models with:
- Accuracy
- ROC AUC score
- Classification report (precision, recall, F1-score)

---

## 🔬 Feature Engineering

- Feature importance was computed for tree-based models.
- The **top 10 most important features** were extracted and used to **retrain models** for comparison.

---

## 📊 Evaluation Metrics

| Model           | Accuracy | ROC AUC | Training Time (s) |
|----------------|----------|---------|--------------------|
| Decision Tree  | 0.95     | 0.94    | 0.0078             |
| Random Forest  | 0.97     | 0.997   | 0.316              |
| XGBoost        | 0.97     | 0.997   | 0.337              |

---

## 📈 Visualizations

- ROC Curve comparison
- Feature importance bar plots for:
  - Decision Tree
  - Random Forest
  - XGBoost

<p align="center">
  <img src="assets/roc_curve.png" alt="ROC Curve" width="500"/>
</p>

<p align="center">
  <img src="assets/feature_importance_xgb.png" alt="Feature Importance - XGBoost" width="500"/>
</p>

---

## ⚙️ Hyperparameter Tuning

We used `GridSearchCV` to search optimal parameters for each model.

- **Decision Tree**: `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`, `gamma`, `reg_lambda`, etc.

---


