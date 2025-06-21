# 🩺 Diabetes Prediction

This project leverages **Data Science** and **Machine Learning** to build a model that predicts whether a person is diabetic based on medical attributes like glucose level, BMI, blood pressure, and more. It includes data preprocessing, feature analysis, model training, and evaluation using Python and scikit-learn.

---

## 📖 Overview

This is a complete **Data Science pipeline** project that:

* Loads and cleans a diabetes dataset
* Performs exploratory data analysis (EDA)
* Applies **feature scaling and transformation**
* Trains a **Logistic Regression** model
* Evaluates its performance using accuracy and confusion matrix

> It’s a beginner-friendly yet complete classification project demonstrating a real-world use case of **predictive healthcare analytics**.

---

## 📊 Dataset

* Source: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* Features:

  * Pregnancies
  * Glucose
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * BMI
  * Diabetes Pedigree Function
  * Age
  * Outcome (Target variable: 1 = Diabetic, 0 = Non-diabetic)

---

## 🛠 Technologies Used

* Python 3
* pandas, numpy
* matplotlib, seaborn
* scikit-learn (LogisticRegression, train\_test\_split, StandardScaler)

---

## 🔁 Workflow

1. **Data Exploration & Cleaning**
2. **Feature Scaling (StandardScaler)**
3. **Train/Test Split**
4. **Model Training (Logistic Regression)**
5. **Prediction & Evaluation**
6. **Result Visualization**

---

## 🧠 Model Used

A **Logistic Regression** model was chosen due to its simplicity and effectiveness for binary classification tasks like predicting diabetes (yes/no).

---

## 📈 Evaluation Metrics

* **Accuracy Score**: Measures the percentage of correct predictions.
* **Confusion Matrix**: Shows how many instances were correctly/incorrectly classified.

> These metrics are printed and interpreted in the notebook after prediction.

---

## 🏃‍♀️ How to Run

1. **Clone the repo**:

   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

   * Open and run `diabetes_prediction.ipynb`

---

## 📊 Results

* The model achieved a decent accuracy on the test set.
* Insights from EDA and correlation heatmaps are visualized using Seaborn.
* The scaled features helped improve model stability.

---

## 🚀 Future Work

* Add more models (Random Forest, XGBoost, etc.) for comparison
* Perform hyperparameter tuning with GridSearchCV
* Handle missing/zero values more robustly
* Deploy the model via Streamlit or Flask

---

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
The model is trained on a limited public dataset and is not approved for clinical or diagnostic use.
Always consult a qualified healthcare provider for any medical concerns.
