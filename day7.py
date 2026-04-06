import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Namma Data (Hours vs Result)
# 0 = Fail, 1 = Pass
hours = np.array([1, 1.5, 2, 4, 5, 6]).reshape(-1, 1) 
result = np.array([0, 0, 0, 1, 1, 1]) # Mudhal 3 peru fail, adutha 3 peru pass

# 2. Creating the ML Model (Classifier)
model = LogisticRegression()

# 3. Training the Model (Padikka vekkirom)
model.fit(hours, result)

# 4. Prediction (Test panrom)
# Oru student 3 hours padichaara, avar pass ah fail ah?
new_student = np.array([[3]]) 
prediction = model.predict(new_student)

print("--- ML Classification Result ---")
if prediction[0] == 1:
    print("Prediction: The student will PASS! 🎉")
else:
    print("Prediction: The student will FAIL. 😢")