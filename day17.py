from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
X = np.array([
    [30, 40], [40, 50], [80, 90], [90, 80], [50, 50], 
    [60, 60], [20, 20], [95, 95], [45, 55], [85, 85]
])
y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1]) 
model = DecisionTreeClassifier(random_state=42)
parameters = {
    'max_depth': [1, 2, 3, 4, 5],
    'criterion': ['gini', 'entropy']
}
print("Searching for the BEST settings... 🔍")
tuner = GridSearchCV(model, parameters, cv=2) 
tuner.fit(X, y)
print("\n--- Hyperparameter Tuning Results ---")
print(f"Best Settings Found: {tuner.best_params_} 🎯")
print(f"Best Accuracy: {tuner.best_score_ * 100}%")

# Importing GridSearchCV, a tool that automatically tests different model settings to find the best ones
# Setting up our dataset with 10 examples, each having two numerical features
# The target labels for our data (0 for one category, 1 for another)
# Creating a standard Decision Tree model to act as our base AI
# Defining the 'Grid': a list of different settings (hyperparameters) we want to test
# 'max_depth' controls how deep the tree grows, 'criterion' is the math it uses to make splits
# Setting up the tuner: It will test every combination of our parameters using cross-validation (cv=2)
# Running the search: The AI trains itself multiple times, trying every single setting combination!
# Printing the absolute best settings it found and the highest accuracy it achieved