# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Example dataset
# Hours Studied (independent variable)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# Exam Scores (dependent variable)
Y = np.array([51, 55, 60, 68, 72, 75, 78, 82, 88, 90])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model with the training data
model.fit(X_train, Y_train)

# Predict values for the testing data
Y_pred = model.predict(X_test)

# Calculate the mean squared error for the test data
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Predict values for the given range of hours studied
X_predict = np.linspace(1, 10, 100).reshape(-1, 1)  # 100 points for a smooth line
Y_predict = model.predict(X_predict)

# Plotting the actual data points
plt.scatter(X_train, Y_train, color='blue', label='Training data')
plt.scatter(X_test, Y_test, color='green', label='Testing data')

# Plotting the regression line
plt.plot(X_predict, Y_predict, color='red', linewidth=2, label='Regression line')

# Adding labels and title
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score (Linear Regression)')

# Adding a legend
plt.legend()

# Display the plot
plt.show()