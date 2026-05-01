from sklearn.svm import SVC
import numpy as np
X = np.array([
    [0, 0],  
    [1, 0],  
    [5, 4], 
    [6, 5],  
    [0, 1],  
    [8, 6]   
])
y = np.array([0, 0, 1, 1, 0, 1])
model = SVC(kernel='linear')
model.fit(X, y)
new_message = np.array([[4, 3]])
prediction = model.predict(new_message)
print("--- SVM Spam Detector ---")
if prediction[0] == 1:
    print("🚨 Alert: This is a SPAM message! (Block it)")
else:
    print("✅ Safe: This is a NORMAL message.")
    
# Importing the Support Vector Machine tool, which draws the best possible dividing line between categories
# Setting up past message data: Let's assume the numbers represent [Links, Suspicious Words]
# The labels for those messages: 1 means it is SPAM, 0 means it is SAFE
# Creating a linear SVM model, telling it to find a straight boundary line to separate the two groups
# Training the model by plotting the messages and finding the perfect dividing line (the "hyperplane")
# Analyzing a brand new message that has 4 links and 3 suspicious words
# Predicting which side of the boundary line this new message lands on
# Printing the final alert if the model's prediction classifies the message as spam