🥗 NutriClass: Food Classification Using Nutritional Data
📌 Project Overview

NutriClass is a machine learning project that classifies food items into multiple categories using nutritional and contextual features such as calories, proteins, fats, carbohydrates, meal type, and preparation method. The project demonstrates an end-to-end machine learning workflow, from data preprocessing to model evaluation.

🎯 Problem Statement

With increasing awareness of nutrition and diet planning, automatically classifying food items based on their nutritional attributes is highly valuable. The goal of this project is to build a robust multi-class classification model that accurately identifies food categories using tabular nutritional data.

🧾 Dataset

Type: Tabular (Synthetic, imbalanced)

Features:

Calories, Protein, Fat, Carbs, Sugar, Fiber

Sodium, Cholesterol, Glycemic Index

Water Content, Serving Size

Meal Type, Preparation Method

Is Vegan, Is Gluten Free

Target Variable: Food_Name

Key Challenge: Class imbalance across food categories

🔍 Project Workflow
1. Data Understanding & Exploration

Dataset inspection and structure analysis

Class distribution analysis to identify imbalance

2. Data Preprocessing

Missing value handling

Duplicate removal

Outlier treatment using IQR method

Encoding categorical variables

Feature scaling and imputation

Stratified train-test split

3. Feature Engineering & Selection

Correlation analysis

Feature importance using Random Forest

Optional dimensionality reduction using PCA

4. Model Training & Comparison

The following machine learning models were trained and compared:

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Gradient Boosting Classifier

5. Evaluation & Results

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

Confusion Matrix

Sample actual vs predicted food labels

📊 Model Evaluation

Due to class imbalance, weighted precision, recall, and F1-score were used as primary evaluation metrics instead of accuracy alone. Ensemble-based models demonstrated superior performance in capturing non-linear relationships between nutritional features.

🧠 Key Insights

High-calorie and high-fat foods such as Pizza and Burger are easier to classify.

Healthier foods like Salad, Apple, and Banana show overlapping nutritional patterns, leading to some misclassification.

Tree-based ensemble models performed best due to their ability to handle feature interactions.

🛠️ Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Environment: Google Colab

📁 Project Structure
NutriClass/
│
├── NutriClass_Food_Classification_FINAL.ipynb
├── NutriClass_Food_Classification_Project_Report.pdf
├── data/
│   └── synthetic_food_dataset_imbalanced.csv
├── README.md
└── requirements.txt

🏁 Conclusion

This project successfully demonstrates the application of machine learning techniques to food classification using nutritional data. The approach can be extended to real-world applications such as dietary recommendation systems, food tracking apps, and health monitoring platforms.

🚀 Future Improvements

Apply SMOTE or class-weighting to improve minority class prediction

Hyperparameter tuning using GridSearchCV

Deploy the model using Streamlit or Flask

Integrate real-world food datasets

👤 Author

Soundarya raju
Senior Project Controls Engineer | Aspiring Data & ML Practitioner

⭐ Acknowledgment

This project was developed as part of a hands-on learning initiative to strengthen practical machine learning and data science skills.
