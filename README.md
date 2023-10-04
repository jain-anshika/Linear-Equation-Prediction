# Machine Learning for Linear Equation Prediction

## Assignment Description

In this assignment, you are tasked with developing a machine learning model that can predict linear equations based on a given dataset. The dataset consists of pairs of input variables and corresponding output values, representing different linear equations. Your goal is to train a model that can accurately predict the coefficients of a linear equation given new input values.

## Requirements

### Dataset Generation

- Generate a dataset of multiple linear equations, each represented by a pair of input variables and the corresponding output value.
- Ensure that the dataset includes a variety of linear equations with different slopes and intercepts.
- You can use a random generator or predefined equations to create the dataset.

### Data Pre-processing

- Perform any necessary pre-processing steps on the dataset, such as data normalization or standardization, to ensure better model performance.

## Implementation

### Dataset Generation

We generated a dataset with a variety of linear equations using Python. The dataset includes random input variables `X1` and `X2`, and the corresponding output variable `y`. We have incorporated different slopes and intercepts to provide a diverse set of linear relationships.

### Model Training and Prediction

We trained a linear regression model using the generated dataset to predict linear equations based on input values `X1` and `X2`. The model provides coefficients for `X1` and `X2`, as well as an intercept and the predicted output `y`.

## Installation
To use this project, follow these steps:

1. Clone the repository to your local machine:
'''
git clone <repository-url>
'''

2. Navigate to the project directory:
'''
cd <project-directory>
'''

3. Install the required Python packages using pip:
'''
pip install pandas numpy matplotlib scikit-learn
'''
4. Run the dataset generation script to generate the dataset:
'''
python generate_dataset.py
'''

5. Run the model script to train the linear regression model and make predictions:
'''
python linear_regression_model.py
'''

Follow these steps to set up and use the project successfully.

## Features
- It is trained on 5000 dataset points.
- It predicts coefficient (a,b) as well as dependent variable (y).
- It plots the graph of the equation.

## Conclusion

This assignment demonstrates the generation of a diverse dataset of linear equations and the training of a linear regression model for predicting coefficients based on input values. By following the requirements and implementing the code, you can accurately predict linear equations and understand their relationships.
