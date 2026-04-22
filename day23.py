import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 1. Load Data (Scikit-Learn kulla irukkura default images!)
# Ithu 8x8 pixels la irukkura kaiyala ezhuthuna numbers (0 to 9)
digits = datasets.load_digits()

# Image-ah computer ku puriyura mathiri flatten panrom (64 numbers in one row)
X = digits.data
y = digits.target # Actual answer

# 2. Train / Test Split (Day 9 trick!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and Train the AI Brain
print("Training the Computer Vision AI... 👁️")
# SVM algorithm use panrom
model = SVC(gamma=0.001)
model.fit(X_train, y_train)

# 4. AI-kitta oru Puthu Image-ah kaati Test panrom!
test_image = X_test[0] # Test set-la irukkura mudhal image
true_answer = y_test[0] # Athoda unmaiyana answer

prediction = model.predict([test_image])

print("\n--- AI Vision Result ---")
print(f"AI Predicts this number is: {prediction[0]} 🎯")
print(f"Actual Number was: {true_answer}")

# 5. Nammalum antha image-ah paapom (using Matplotlib)
plt.imshow(test_image.reshape(8, 8), cmap=plt.cm.gray_r)
plt.title(f"AI Predicted: {prediction[0]}")
plt.axis('off') # Scale thevai illai
plt.show()