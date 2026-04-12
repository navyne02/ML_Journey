from sklearn.neural_network import MLPClassifier
import numpy as np

# 1. Namma Data 
# Columns: [Study Hours, Sleep Hours]
X = np.array([
    [2, 4],  # Low study, low sleep -> Fail
    [8, 8],  # High study, high sleep -> Pass
    [9, 2],  # High study, low sleep -> Fail (Burnt out!)
    [3, 9]   # Low study, high sleep -> Fail (Lazy!)
])

# 0 = Fail, 1 = Pass
y = np.array([0, 1, 0, 0])

# 2. Creating the Artificial Brain (Neural Network)
# hidden_layer_sizes=(4,) na 4 hidden neurons irukku nu artham
model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, random_state=42)

# 3. Training the Brain
print("Training the Neural Network... 🧠")
model.fit(X, y)

# 4. Test with a New Student
# Student stats -> Study: 7 hours, Sleep: 7 hours
new_student = np.array([[7, 7]])
prediction = model.predict(new_student)

print("--- Neural Network Prediction ---")
if prediction[0] == 1:
    print("AI Brain says: Student will PASS! 🎓✨")
else:
    print("AI Brain says: Student will FAIL. ⚠️ Needs balance!")