from sklearn.neighbors import KNeighborsClassifier
import numpy as np
X = np.array([
    [20, 80],   
    [15, 85],   
    [300, 10],  
    [400, 5],  
    [25, 75],  
    [350, 15]  
])
y = np.array([0, 0, 1, 1, 0, 1])
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
new_player = np.array([[250, 12]])
prediction = model.predict(new_player)
print("--- KNN Player Prediction ---")
if prediction[0] == 1:
    print("AI Predicts: This player is a SNIPER! 🎯")
else:
    print("AI Predicts: This player is an ASSAULTER! 🔫")
    
# Importing the K-Nearest Neighbors tool, which classifies data based on its closest neighbors
# Setting up player stats: [Engagement Distance, Fire Rate]
# The player classes: 1 means SNIPER (long range, slow fire), 0 means ASSAULTER (close range, fast fire)
# Creating the KNN model and telling it to check the 3 closest neighbors (n_neighbors=3)
# Training the model by plotting all the known players onto an invisible map
# Setting up a brand new player with an engagement distance of 250 and fire rate of 12
# Asking the model to predict the class by taking a majority vote of the 3 nearest neighbors
# Printing the final prediction based on the results of that neighbor vote