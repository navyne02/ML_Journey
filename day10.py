from sklearn.tree import DecisionTreeClassifier
import numpy as np
X = np.array([
    [90, 4], 
    [60, 5], 
    [85, 2], 
    [95, 6],  
])
y = np.array([1, 0, 0, 1])
model = DecisionTreeClassifier()
model.fit(X, y)
new_student = np.array([[88, 4]])
prediction = model.predict(new_student)
print("--- Decision Tree Prediction ---")
if prediction[0] == 1:
    print("Result: Student will PASS! 🎓")
else:
    print("Result: Student will FAIL. 📚 Try harder!")
    
# Importing the Decision Tree tool, which makes predictions by creating a hidden flowchart of rules
# Setting up our data: Each student has two features (e.g., [Test Score, Study Hours])
# The results for those past students: 1 means PASS, 0 means FAIL
# Creating the blank Decision Tree model
# Training the model so it can analyze the data and build its internal yes/no rulebook
# Asking the model to predict the result for a new student with a score of 88 and 4 hours of study
# Printing a celebratory message if they pass, or an encouraging message if they fail