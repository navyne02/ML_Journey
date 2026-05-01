import numpy as np
from sklearn.linear_model import LogisticRegression
hours = np.array([1, 1.5, 2, 4, 5, 6]).reshape(-1, 1) 
result = np.array([0, 0, 0, 1, 1, 1])
model = LogisticRegression()
model.fit(hours, result)
new_student = np.array([[3]]) 
prediction = model.predict(new_student)
print("--- ML Classification Result ---")
if prediction[0] == 1:
    print("Prediction: The student will PASS! 🎉")
else:
    print("Prediction: The student will FAIL. 😢")
    
# Numpy and Scikit-Learn provide the tools to build our machine learning model
# Creating our dataset to link study hours with the final exam results (0 for fail, 1 for pass)
# Creating and training a Logistic Regression model so it learns the exact cutoff point for passing
# Predicting the result for a new student studying 3 hours and printing whether they PASS or FAIL