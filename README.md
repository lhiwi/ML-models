# Machine Learning Assignment: Genetic Disorder Prediction
Overview
This project involves building and deploying a machine learning model to predict genetic disorders based on patient demographics and medical features. The approach emphasizes accuracy and balanced performance across all classes using Decision Trees combined with techniques like hyperparameter tuning and class balancing.

Table of Contents
Problem Definition
Dataset
Exploratory Data Analysis (EDA)
Feature Engineering
Model Training and Tuning
Deployment
Usage
Results and Evaluation
Future Work
1. Problem Definition
Objective
To develop a predictive model that determines whether a patient has a genetic disorder using relevant medical and demographic features.

Success Metrics
Achieve a model accuracy of over 80%.
Optimize precision, recall, and F1-score for balanced predictions.
2. Dataset
Source
The dataset is sourced from Kaggle: Predict the Genetic Disorder Dataset.

Key Features
Demographics: Age, gender, parental data.
Medical Information: Test results, symptoms, and medical history.
Target Variable: Indicates the presence of a genetic disorder.
3. Exploratory Data Analysis (EDA)
Steps Undertaken:
Class Distribution: Examined to identify target imbalance.
Feature Visualization:
Used box plots for numerical features to understand distribution.
Generated correlation heatmaps for feature interdependencies.
Insights:
Identified significant imbalance in the target classes.
Found a moderate correlation between parental ages.
Addressed outliers in blood test results with imputation techniques.
4. Feature Engineering
Handling Missing Data:
Numerical Columns: Filled missing values with the median.
Categorical Columns: Imputed with the mode or domain-specific replacements.
Encoding Techniques:
Converted boolean features (e.g., symptoms) to binary (0/1).
Applied one-hot encoding for multi-class categorical variables.
Balancing Classes:
Used class weights in the Decision Tree model to manage imbalances.
Tested SMOTE (Synthetic Minority Oversampling Technique) for oversampling minority classes.
Normalization:
Scaled numerical features using StandardScaler to maintain consistency.
5. Model Training and Tuning
Algorithm
A Decision Tree Classifier was chosen for its ability to handle both categorical and numerical data effectively while offering interpretability.

Hyperparameter Optimization
Performed using GridSearchCV (5-fold cross-validation) on these parameters:

criterion (gini/entropy)
max_depth
min_samples_split
min_samples_leaf
Final Model Configuration:
Criterion: Entropy
Max Depth: 5
Min Samples Split: 2
Min Samples Leaf: 1
6. Deployment
Steps to Deploy:
Model Serialization: Saved the trained model with joblib:

python
Copy code
joblib.dump(best_dt_model, 'decision_tree_model_best_tuned.pkl')
Flask API:

Developed a RESTful API with the following endpoints:
/: Home route with a welcome message.
/predict: Accepts JSON input and returns predictions.
Testing:

Locally tested API with Postman and cURL.
Deployment Preparation:

Created a requirements.txt file for dependencies.
Designed a Procfile to host the API on platforms like Heroku.