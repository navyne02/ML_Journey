from sklearn.neural_network import MLPClassifier
import numpy as np
X = np.array([
    [2, 4],  
    [8, 8],  
    [9, 2],  
    [3, 9]   
])
y = np.array([0, 1, 0, 0])
model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000, random_state=42)
print("Training the Neural Network... 🧠")
model.fit(X, y)
new_student = np.array([[7, 7]])
prediction = model.predict(new_student)
print("--- Neural Network Prediction ---")
if prediction[0] == 1:
    print("AI Brain says: Student will PASS! 🎓✨")
else:
    print("AI Brain says: Student will FAIL. ⚠️ Needs balance!")
    
# Importing the Neural Network tool, which mimics how biological brain cells process information
# Setting up the student data: Two distinct features (e.g., [Study Hours, Sleep Hours])
# The final results for these past students: 1 means PASS, 0 means FAIL
# Creating the 'Brain': We give it 1 hidden layer with 4 'neurons' and let it study up to 2000 times (max_iter)
# Training the Neural Network: The model repeatedly adjusts its internal connections to learn the hidden patterns
# Setting up a brand new student with scores of 7 and 7
# Asking the trained network to process these scores through its hidden neurons to predict the final outcome
# Printing the final prediction based on the network's complex calculations