# Product-Category-Classifier

This repository contains a machine learning model for classifying products into categories. The project includes data preprocessing, model training, evaluation, and deployment.

## Features

- **End-to-End Pipeline:** Covers data cleaning, feature engineering, model selection, and deployment.
- **Multi-Category Support:** Suitable for e-commerce and inventory management.
- **Scalable Model:** Easily extendable to more categories or features.
- **Comprehensive Documentation:** Includes Jupyter notebooks for EDA and model training.

## Installation
Install dependencies:
pip install -r requirements.txt
Usage
Running the Models
Random Forest Model:

Run the random_forest_model.py script to train and test the Random Forest model.
bash
Copy code
python random_forest_model.py
Logistic Regression Model:

Run the logistic_regression_model.py script to train and test the Logistic Regression model.
bash
Copy code
python logistic_regression_model.py
Data
The dataset is included in the data/ directory. The scripts are configured to load the data from this directory. If using your data, replace the existing file with your dataset, ensuring the format remains consistent.

Model Overview
Random Forest: A robust ensemble method used for classification, handling large datasets with high accuracy.
Logistic Regression: A straightforward, yet powerful algorithm for binary and multi-class classification tasks.
Results
The performance of each model, including accuracy and other metrics, will be displayed in the console after running the scripts.
