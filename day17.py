from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. Namma Data (Student Marks vs Pass/Fail)
# [Maths, Science]
X = np.array([
    [30, 40], [40, 50], [80, 90], [90, 80], [50, 50], 
    [60, 60], [20, 20], [95, 95], [45, 55], [85, 85]
])
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1]) 

# 2. Base Model (Normal AI)
model = DecisionTreeClassifier(random_state=42)

# 3. Tuning Options (AI kitta intha settings ellam try panna solrom)
# max_depth = Tree evlo aazhama kelvigal kekkanum
parameters = {
    'max_depth': [1, 2, 3, 4, 5],
    'criterion': ['gini', 'entropy']
}

# 4. The Tuner (GridSearchCV)
print("Searching for the BEST settings... 🔍")
# cv=2 means athuve data-va 2 ah pirichu test panni paakum (Cross Validation)
tuner = GridSearchCV(model, parameters, cv=2) 
tuner.fit(X, y)

# 5. Results
print("\n--- Hyperparameter Tuning Results ---")
print(f"Best Settings Found: {tuner.best_params_} 🎯")
print(f"Best Accuracy: {tuner.best_score_ * 100}%")