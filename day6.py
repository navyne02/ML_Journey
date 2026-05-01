import numpy as np
from sklearn.linear_model import LinearRegression
hours = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) 
marks = np.array([20, 40, 60, 80, 100]) 
model = LinearRegression()
model.fit(hours, marks)
new_hours = np.array([[6]]) 
predicted_marks = model.predict(new_hours)
print("--- My First ML Prediction ---")
print(f"If you study for 6 hours, predicted marks: {predicted_marks[0]:.2f}")

# The Setup: It first gives the computer a set of past examples, showing how many hours of study led to certain marks.
# The Brain: It creates a fresh, blank machine learning model, which acts like a new brain waiting to be taught.
# The Learning: It feeds those past examples into the model so it can study the patterns and learn the exact connection between time and marks.
# The Prediction: It uses what it just learned to guess the final score someone would get if they studied for six hours, and prints the result.