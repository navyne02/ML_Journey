from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. Namma Data 
# Columns: [Attendance %, Study Hours]
X = np.array([
    [90, 4],  # 90% attendance, 4 hours -> Pass
    [60, 5],  # 60% attendance, 5 hours -> Fail (Attendance low)
    [85, 2],  # 85% attendance, 2 hours -> Fail (Study hours low)
    [95, 6],  # 95% attendance, 6 hours -> Pass
])

# 0 = Fail, 1 = Pass
y = np.array([1, 0, 0, 1])

# 2. Creating the ML Model (The Tree)
model = DecisionTreeClassifier()

# 3. Training the Model
model.fit(X, y)

# 4. Test with a New Student
# Attendance = 88%, Study Hours = 4
new_student = np.array([[88, 4]])
prediction = model.predict(new_student)

print("--- Decision Tree Prediction ---")
if prediction[0] == 1:
    print("Result: Student will PASS! 🎓")
else:
    print("Result: Student will FAIL. 📚 Try harder!")