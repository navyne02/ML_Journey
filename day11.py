from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Namma Data 
# Columns: [Maths Mark, Science Mark]
X = np.array([
    [40, 50],  # Fail
    [80, 90],  # Pass
    [90, 85],  # Pass
    [30, 20],  # Fail
    [60, 60],  # Border Pass
    [95, 95]   # Pass
])

# 0 = Fail, 1 = Pass
y = np.array([0, 1, 1, 0, 1, 1])

# 2. Creating the ML Model (A forest with 10 Trees!)
# n_estimators na ethaana trees venum nu artham
model = RandomForestClassifier(n_estimators=10, random_state=42)

# 3. Training the Forest
model.fit(X, y)

# 4. Test with a New Student
# Student marks -> Maths: 75, Science: 80
new_student = np.array([[75, 80]])
prediction = model.predict(new_student)

print("--- Random Forest Prediction ---")
if prediction[0] == 1:
    print("Forest says: Student will PASS! 🌲✅")
else:
    print("Forest says: Student will FAIL. 🍂❌")