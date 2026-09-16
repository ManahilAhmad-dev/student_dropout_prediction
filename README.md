# Student Dropout Prediction

## Project Overview

This project uses Machine Learning to predict whether a student may be at risk of dropping out.

A Logistic Regression model is trained using academic, demographic, socioeconomic, behavioral, and enrollment-related information.

## Objective

The goal is to provide an early-warning system that can help educational institutions identify students who may need additional academic, financial, or counseling support.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Gradio
- Google Colab
- GitHub

## Machine Learning Workflow

1. Problem Understanding
2. Dataset Collection & Exploration
3. Data Cleaning & Preprocessing
4. Exploratory Data Analysis
5. Logistic Regression Model
6. Model Evaluation
7. Student Risk Prediction Application
8. Documentation & Deployment

## Model

The project uses Logistic Regression because the target variable is binary:

- 0 = Did not drop out
- 1 = Dropped out

The model also provides a dropout probability that is used to display a risk category.

## Risk Categories

- Below 30% → Low Risk
- 30%–59% → Medium Risk
- 60% or above → High Risk

## Application

The Gradio application allows users to enter student information and receive:

- Dropout probability
- Risk category
- Prediction result

## Dataset

Dataset:
https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset

## Potential Use

The application can serve as an educational early-warning tool. Educational institutions could use predictions to identify students who may need additional support and connect them with appropriate academic, financial, or counseling resources.

The prediction should be treated as a support tool rather than a final decision about a student.

## Author

Manahil Ahmad

Data Science Student
