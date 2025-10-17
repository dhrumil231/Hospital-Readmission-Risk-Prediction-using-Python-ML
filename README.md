-----

# 🏥 Hospital Readmission Risk Prediction using Python & ML

A machine learning project designed to predict the **risk of hospital readmission** within a 30-day window. This solution leverages Python's data science stack to process complex healthcare data, employ advanced feature selection, and evaluate multiple classification models to identify high-risk patients.

| 🎯 **Goal** | 🐍 **Language** | 🧠 **Model Type** | 📊 **Best Metric** |
| :---: | :---: | :---: | :---: |
| **Risk Prediction** | **Python** | **Classification** | **Accuracy $\approx 49.79\%$ (SVM)** |

-----

## ⭐ Project Significance & Objectives

Preventing unnecessary hospital readmissions is critical for improving patient care and reducing healthcare costs. This project aims to provide a reliable, data-driven tool to help clinical staff intervene effectively.

The core objectives include:

  * **Identifying High-Risk Patients:** Accurately classifying patients who are likely to be readmitted within 30 days.
  * **Feature Importance:** Determining which **clinical and demographic factors** are the strongest predictors of readmission risk.
  * **Model Comparison:** Implementing and comparing a range of supervised learning models to find the **most robust and clinically useful predictor**.

-----

## 🛠️ Methodology & Machine Learning Workflow

The project follows a rigorous data science workflow documented in the main notebook.

1.  **🔍 Data Exploration & Cleaning:** Initial data loading, handling missing values, and addressing potential feature errors.
2.  **⚙️ Advanced Feature Selection:** Employing advanced algorithms, such as **Boruta**, to select the **most impactful features** for the predictive models.
3.  **🧠 Comprehensive Model Training:** Implementing and training a diverse set of supervised learning models:
      * **Gaussian Naive Bayes**
      * **Decision Tree Classifier**
      * **Random Forest Classifier**
      * **Support Vector Machines (SVM)**
      * **Neural Networks (MLP)**
      * **AdaBoost Classifier**
      * **Gradient Boosting Classifier**
4.  **📈 Performance Analysis:** Evaluating all models using key classification metrics to identify the most effective solution. This also included the use of **Imblearn** for handling imbalanced datasets.

-----

## 📊 Results & Best Performer

After comprehensive evaluation, the **Support Vector Machines (SVM)** model was found to be the most promising baseline predictor:

| Metric | SVM Model Score |
| :--- | :--- |
| **Accuracy** | $\approx 49.79\%$ (on the test set) |
| **Precision, Recall, F1-Score** | Detailed in the classification report and confusion matrix within the notebook. |

The project includes a **detailed classification report** and **confusion matrix** to provide a transparent view of the model's performance on positive and negative classes.

-----

## 💻 Tech Stack & Dependencies

The entire project is built using Python and its standard data science libraries:

  * **Language:** **Python**
  * **Core Libraries:**
      * **[Pandas](https://pandas.pydata.org/):** For data manipulation and analysis.
      * **[NumPy](https://numpy.org/):** For numerical operations.
      * **[Scikit-learn](https://scikit-learn.org/stable/):** For core machine learning models and utilities.
      * **[Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/):** For data visualization.
      * **[Imblearn](https://imbalanced-learn.org/stable/):** Specifically used for handling the **imbalanced nature** of healthcare readmission datasets.
      * **[Boruta](https://github.com/scikit-learn-contrib/boruta_py):** Used for advanced feature selection.

-----

## ⚙️ Getting Started (Run Locally)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dhrumil231/Hospital-Readmission-Risk-Prediction-using-Python-ML.git
    ```
2.  **Navigate to the directory:**
    ```bash
    cd Hospital-Readmission-Risk-Prediction-using-Python-ML
    ```
3.  **Install dependencies:**
    ```bash
    # Install all libraries listed in the Tech Stack section.
    # A requirements.txt file would be recommended for easy setup.
    pip install pandas numpy scikit-learn matplotlib seaborn imblearn boruta
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  **Run the analysis:** Open the main analysis notebook to execute the full data-to-prediction pipeline.

-----

## 🔮 Future Work

  * **Deep Learning Optimization:** Fine-tuning the Neural Networks (MLP) and exploring more advanced architectures like CNNs or LSTMs for time-series features.
  * **Hyperparameter Tuning:** Implementing systematic hyperparameter optimization (e.g., Grid Search, Bayesian Optimization) for the top-performing models (e.g., SVM, Random Forest).
  * **Explainable AI (XAI):** Integrating tools like **SHAP or LIME** to provide clinical interpretability to the model's predictions.
