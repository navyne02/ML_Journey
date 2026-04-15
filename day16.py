import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Simple Data & Model Training
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
y = np.array([0, 0, 1, 1])

model = RandomForestClassifier()
model.fit(X, y)
print("1. AI Model Trained Successfully! 🧠")

# 2. Saving the Model (Save Game 💾)
# Intha line thaan unga AI-ah oru file-ah mathum!
joblib.dump(model, 'my_saved_ai.pkl')
print("2. Model Saved as 'my_saved_ai.pkl' 📦")
print("-" * 30)

# ==========================================
# Imagine you closed the laptop and came back the next day...
# ==========================================

# 3. Loading the Model (Load Game 🎮)
loaded_ai = joblib.load('my_saved_ai.pkl')
print("3. Model Loaded from file successfully! ♻️")

# 4. Using the loaded model
new_data = np.array([[7, 8]])
prediction = loaded_ai.predict(new_data)

print(f"4. Prediction from Loaded AI: {prediction[0]} ✅")