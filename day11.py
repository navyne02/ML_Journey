from sklearn.ensemble import RandomForestClassifier
import numpy as np
X = np.array([
    [40, 50],  
    [80, 90],  
    [90, 85],  
    [30, 20],  
    [60, 60],  
    [95, 95]   
])
y = np.array([0, 1, 1, 0, 1, 1])
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)
new_student = np.array([[75, 80]])
prediction = model.predict(new_student)
print("--- Random Forest Prediction ---")
if prediction[0] == 1:
    print("Forest says: Student will PASS! 🌲✅")
else:
    print("Forest says: Student will FAIL. 🍂❌")
    
# Importing the Random Forest tool, which creates a "committee" of multiple Decision Trees
# Setting up the past data: Each student has two scores (e.g., [Math Score, Science Score])
# The final results for those students: 1 means PASS, 0 means FAIL
# Creating the model and telling it to build exactly 10 separate decision trees (n_estimators=10)
# Training the model so all 10 trees can study the data and build their own rulebooks
# Setting up the scores for a brand new student: 75 and 80
# Asking the forest to predict the result. The 10 trees will take a vote, and majority wins!
# Printing the final verdict based on the forest's majority vote