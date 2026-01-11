House Price Prediction using Linear Regression
Project Overview

This project implements a Linear Regression model to predict house prices based on:
Square footage of the house
Number of bedrooms
Number of bathrooms
The goal is to build a simple and interpretable baseline regression model using real-world housing data.

Dataset

Source: Kaggle – House Prices: Advanced Regression Techniques

File Used: train.csv

Target Variable: SalePrice

Features Selected
Feature Name	Description
GrLivArea	Above-ground living area (square feet)
BedroomAbvGr	Number of bedrooms above ground
FullBath	Number of full bathrooms
Tech Stack

Language: Python 3

Libraries:
pandas
numpy
scikit-learn

Project Structure
house_price_project/
│── task1.py
│── train.csv
│── README.md
│── .venv/

Model Used

Algorithm: Linear Regression

Reason: Simple, fast, and easy to interpret
Workflow
Load the dataset
Select relevant features

Handle missing values using median imputation
Split data into training and testing sets (80–20)
Scale features using StandardScaler

Train Linear Regression model
Evaluate using MAE, RMSE, and R² score

Evaluation Metrics
MAE (Mean Absolute Error): Average prediction error
RMSE (Root Mean Squared Error): Penalizes large errors
R² Score: Measures how well the model explains variance

How to Run the Project
1. Activate Virtual Environment
.\.venv\Scripts\activate

2. Install Dependencies
python -m pip install pandas numpy scikit-learn

3. Run the Program
python task1.py

Sample Output
Model Evaluation Results
------------------------
Mean Absolute Error (MAE): 22000.45
Root Mean Squared Error (RMSE): 34000.12
R2 Score: 0.68

Results Interpretation:
Larger houses (higher square footage) significantly increase predicted price
Bathrooms have a stronger impact than bedroom count
The model provides a solid baseline but does not capture non-linear relationships

Limitations:
Uses only three features
Assumes linear relationships
Does not handle outliers explicitly

Future Improvements:
Add more features such as OverallQual, YearBuilt
Try Ridge or Lasso regression
Perform feature engineering
Use cross-validation
