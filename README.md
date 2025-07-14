# 🤝 Voting Ensemble Classifier with XGBoost & AdaBoost

A powerful ensemble learning system combining **XGBoost** and **AdaBoost**, trained on the **Diabetes dataset** with optimized hyperparameters using **GridSearchCV**. This project demonstrates a clean, scalable approach to binary classification using boosting and model ensembling.

---

## 📌 Project Highlights

- ✅ Uses **XGBoost** and **AdaBoost** classifiers
- 🎯 Optimized using **GridSearchCV**
- ⚖️ Handles **imbalanced class distribution** (500:268)
- 🔍 Final ensemble uses **VotingClassifier** (`hard` voting)
- 📊 Achieves **~77–80% accuracy**, tunable up to 85%+

---

## 📁 Dataset

- **Source:** Pima Indians Diabetes Dataset (`diabetes.csv`)
- **Shape:** `(768, 9)` with 8 features + 1 binary target (`Outcome`)
- **Imbalance:** 500 samples of class 0, 268 of class 1

---

## 🔧 Tools & Libraries

- Python 3.10+
- scikit-learn
- xgboost
- pandas, numpy, matplotlib (for EDA)

---

## 🚀 Model Workflow

1. Load and preprocess dataset
2. Handle imbalance (`scale_pos_weight`)
3. Scale features using `StandardScaler`
4. Train AdaBoost and XGBoost using `GridSearchCV`
5. Build a **VotingClassifier** with weighted hard voting
6. Evaluate using accuracy and classification report

---

## 🧪 Final Accuracy

| Model       | Accuracy (%) |
|-------------|--------------|
| XGBoost     | ~74.0%       |
| AdaBoost    | ~76.0%       |
| 🤝 Voting Ensemble | ✅ **77.27%** (Tuned) |

---

## 🗂️ Folder Structure

📦 VotingEnsemble-Diabetes
├── diabetes.csv
├── ensemble_model.ipynb
├── README.md
└── requirements.txt



---

## 🖥️ Run Locally

1. Clone the repo
```bash
git clone https://github.com/yourusername/Diabetes-prediction.git
cd VotingEnsemble-Diabetes

📌 Future Improvements
Add more classifiers (e.g. Gradient Boosting)

Use soft voting with calibrated probabilities

Perform feature importance analysis

Save models using joblib or pickle
