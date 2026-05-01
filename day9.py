import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total Students: {len(X)}")
print(f"Training Students (for Model to learn): {len(X_train)}")
print(f"Testing Students (for Final Exam): {len(X_test)}\n")
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print("--- AI Final Exam Results ---")
print(f"Actual Answers:    {y_test}")
print(f"AI's Predictions:  {predictions}")
print(f"Model Accuracy:    {score * 100}% 🏆")

# Importing tools for splitting data, building the model, and checking its accuracy
# Setting up our data: 10 students' study hours (X) and whether they passed or failed (y)
# Splitting the data: 80% for the AI to study (train) and 20% kept secret for testing
# Printing out how many students are in the total, training, and testing groups
# Creating the Logistic Regression model and teaching it using ONLY the training data
# Giving the AI the final exam by asking it to predict outcomes for the testing data
# Comparing the AI's predictions to the actual answers to calculate its final score
# Printing the actual answers, the AI's guesses, and its overall accuracy percentage