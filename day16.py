import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
y = np.array([0, 0, 1, 1])
model = RandomForestClassifier()
model.fit(X, y)
print("1. AI Model Trained Successfully! 🧠")
joblib.dump(model, 'my_saved_ai.pkl')
print("2. Model Saved as 'my_saved_ai.pkl' 📦")
print("-" * 30)
loaded_ai = joblib.load('my_saved_ai.pkl')
print("3. Model Loaded from file successfully! ♻️")
new_data = np.array([[7, 8]])
prediction = loaded_ai.predict(new_data)
print(f"4. Prediction from Loaded AI: {prediction[0]} ✅")

# Importing joblib, a tool that allows us to save Python objects directly to a file
# Setting up our basic training data and target labels
# Creating and training the Random Forest model (the "learning" phase)
# Saving the fully trained 'brain' into a file called 'my_saved_ai.pkl'
# Loading the trained AI back out of that file and into a brand new variable
# Creating some new data to test if the loaded AI still remembers its training
# Asking the loaded AI to make a prediction, proving it works without being retrained!