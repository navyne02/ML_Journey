from sklearn.svm import SVC
import numpy as np

# 1. Namma Data (Message Analysis)
# Columns: [Number of Links, Number of 'Urgent/Offer' words]
X = np.array([
    [0, 0],  # Normal text to friend
    [1, 0],  # Normal message with 1 link
    [5, 4],  # Spam (5 links, 4 urgent words)
    [6, 5],  # Spam
    [0, 1],  # Normal
    [8, 6]   # Heavy Spam
])

# 0 = Normal, 1 = Spam
y = np.array([0, 0, 1, 1, 0, 1])

# 2. Creating the SVM Model (kernel='linear' na straight line)
model = SVC(kernel='linear')

# 3. Training the Model
model.fit(X, y)

# 4. Test with a New Message
# Oru puthu message varuthu: 4 links and 3 urgent words irukku
new_message = np.array([[4, 3]])
prediction = model.predict(new_message)

print("--- SVM Spam Detector ---")
if prediction[0] == 1:
    print("🚨 Alert: This is a SPAM message! (Block it)")
else:
    print("✅ Safe: This is a NORMAL message.")