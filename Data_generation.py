import numpy as np
import pandas as pd

def generate_linear_dataset(num_samples, slopes, intercepts):
    np.random.seed(0)
    num_varieties = len(slopes)
    X1 = np.random.rand(num_samples) * 10  
    X2 = np.random.rand(num_samples) * 10  
    variety_indices = np.random.choice(num_varieties, num_samples)  
    y = []

    for i in range(num_samples):
        slope = slopes[variety_indices[i]]
        intercept = intercepts[variety_indices[i]]
        y_i = slope * X1[i] + intercept * X2[i] + np.random.randn()
        y.append(y_i)

    return X1, X2, y

#the dataset with variety in slopes and intercepts
num_samples = 1000
slopes = [2.0, 3.0, 1.5]  
intercepts = [1.0, 0.5, 2.0]  
X1, X2, y = generate_linear_dataset(num_samples, slopes, intercepts)

#dataFrame to store the data
data = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

#making a CSV file of the generated 1000 datasets
data.to_csv('linear_dataset.csv', index=False)

print("Dataset saved as 'linear_dataset.csv'.")
