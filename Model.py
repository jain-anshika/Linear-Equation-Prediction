import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#reading the dataset from csv file
df = pd.read_csv("_linear_dataset.csv")

#using linear regression model
reg = LinearRegression()

#training model 
reg.fit(df[['X1', 'X2']], df['y'])

coefficients = reg.coef_
intercept = reg.intercept_

#prompt from user
x1 = float(input("Enter the value for X1: "))
x2 = float(input("Enter the value for X2: "))

#predicting y wwhen new inputs are entered
predicted_y = reg.predict([[x1, x2]])

x1_range = np.linspace(df['X1'].min(), df['X1'].max(), 100)

predicted_y_range = coefficients[0] * x1_range + coefficients[1] * x2 + intercept

predicted_y = reg.predict([[x1, x2]])

#printing all the coeff. that are predicted and calculated.
print("Coefficient for X1 (a):", reg.coef_[0])
print("Coefficient for X2 (b):", reg.coef_[1])
print("Intercept (c):", reg.intercept_)
print("Predicted y:", predicted_y[0])

#calculating and printing r^2 score.
y_true = df['y']
y_pred = reg.predict(df[['X1', 'X2']])
r2 = r2_score(y_true, y_pred)
print("R-squared score:", r2)

#plotting the graph of generated data sets
plt.scatter(df['X1'], df['y'], label='Generated Data Points', color='lightblue')
#ploting the predicted point
plt.scatter(x1, predicted_y, label='Predicted Point', color='green')
#plot the line graph of the linear equation with multiple variable
plt.plot(x1_range, predicted_y_range, label='Linear Equation', color='purple')

#labelling the graph
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Plot')
plt.grid(True)
#for displaying the grapj
plt.show()


