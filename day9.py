import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Namma Data (10 Students-oda data: Hours Studied vs Pass/Fail)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # 0=Fail, 1=Pass

# 2. Data-va Pirikirom! (80% Padikka, 20% Test panna)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total Students: {len(X)}")
print(f"Training Students (for Model to learn): {len(X_train)}")
print(f"Testing Students (for Final Exam): {len(X_test)}\n")

# 3. Model-ah Train Panrom (Training data mattum thaan kudukurom)
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Exam Time! (Testing data-va predict panna solrom)
predictions = model.predict(X_test)

# 5. Mark Podurom (Accuracy)
score = accuracy_score(y_test, predictions)

print("--- AI Final Exam Results ---")
print(f"Actual Answers:    {y_test}")
print(f"AI's Predictions:  {predictions}")
print(f"Model Accuracy:    {score * 100}% 🏆")