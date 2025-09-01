# Hospital-Readmission-Risk-Prediction-using-Python-ML
Hospital Readmission Risk Prediction using Machine Learning - This project, titled "Hospital Readmission Risk Prediction using Machine Learning," is a data science initiative that leverages machine learning to forecast the likelihood of patients being readmitted to a hospital. Utilizing a comprehensive healthcare dataset, the project aims to identify key factors contributing to readmission and build a predictive model to help healthcare providers improve patient care and resource management.

🏥 Project Overview & Key Features- This project utilizes a structured healthcare dataset to predict hospital readmission rates. The analysis and modeling are contained within a single Jupyter Notebook.

Key features include:
1) Extensive Data Preprocessing: The notebook demonstrates how to handle missing values, clean data, and prepare it for model training.
2) Advanced Feature Engineering: The project applies advanced feature selection techniques, such as Recursive Feature Elimination with Cross-Validation (RFECV) and the Boruta algorithm, to identify the most predictive features.
3) Comprehensive Model Evaluation: A wide range of supervised learning models are implemented and evaluated using key performance metrics like accuracy, precision, recall, and F1-score to determine the best-performing model.

📊 Methodology & Results
The machine learning workflow follows these steps:
1) Data Exploration: Initial data loading and cleaning, including handling missing values and analyzing feature distributions.
2) Feature Selection: Employing advanced algorithms to select the most impactful features for the predictive models.
3) Model Training: Implementing and training a diverse set of supervised learning models, including Decision Trees, Random Forest, Support Vector Machines (SVM), and Neural Networks (MLP).
4) Performance Analysis: Evaluating and comparing the models to identify the most effective solution.

The SVM model was found to be the most promising, achieving an accuracy of approximately 49.79% on the test set. The project includes a detailed classification report and confusion matrix to provide a transparent view of the model's performance on precision, recall, and F1-score for both positive and negative classes.

🛠️ Tech Stack
This project is built using Python and a suite of powerful data science libraries.

Languages: Python

Core Libraries:
1) Pandas for data manipulation and analysis.
2) NumPy for numerical operations.
3) Scikit-learn for machine learning models and utilities.
4) Matplotlib and Seaborn for data visualization.
5) imblearn for handling imbalanced datasets.
6) Boruta for feature selection.

Models:
- Gaussian Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Hist Gradient Boosting Classifier
- Extra Trees Classifier
- Support Vector Classifier (SVC)
- Multilayer Perceptron (MLP) Classifier

📊 Dataset
The project utilizes a structured healthcare dataset containing over 100,000 patient records. The data includes:

Patient Demographics: Age, race, gender.
Encounter Information: Admission type, length of stay, number of lab tests, procedures, and medications.
Clinical Data: Primary and secondary diagnoses, and diabetes-specific metrics (e.g., A1C test results, medication).
Data preprocessing involved handling a large number of missing values and normalizing or scaling numerical features to prepare them for model training.
