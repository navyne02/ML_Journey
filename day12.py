from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. Namma Data (PUBG Player Stats)
# Columns: [Average Target Distance (meters), Movement Speed]
X = np.array([
    [20, 80],   # Close range, high speed -> Assaulter
    [15, 85],   # Assaulter
    [300, 10],  # Long range, low speed -> Sniper
    [400, 5],   # Sniper
    [25, 75],   # Assaulter
    [350, 15]   # Sniper
])

# 0 = Assaulter, 1 = Sniper
y = np.array([0, 0, 1, 1, 0, 1])

# 2. Creating the ML Model 
# n_neighbors (K) = 3 nu set panrom (Check top 3 nearest players)
model = KNeighborsClassifier(n_neighbors=3)

# 3. Training the Model
model.fit(X, y)

# 4. Test with a New Unknown Player
# Distance: 250m, Speed: 12
new_player = np.array([[250, 12]])
prediction = model.predict(new_player)

print("--- KNN Player Prediction ---")
if prediction[0] == 1:
    print("AI Predicts: This player is a SNIPER! 🎯")
else:
    print("AI Predicts: This player is an ASSAULTER! 🔫")