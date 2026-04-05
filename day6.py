import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Namma Data (Hours vs Marks)
# Note: ML model-ku X eppovum 2D array-ah thaan irukkanum
hours = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) 
marks = np.array([20, 40, 60, 80, 100]) 

# 2. Creating the ML Model (Oru puthu brain-ah create panrom)
model = LinearRegression()

# 3. Training the Model (Padikka vekkirom!)
model.fit(hours, marks)

# 4. Prediction (Test panrom)
new_hours = np.array([[6]]) # 6 hours padicha enna aagum?
predicted_marks = model.predict(new_hours)

print("--- My First ML Prediction ---")
print(f"If you study for 6 hours, predicted marks: {predicted_marks[0]:.2f}")