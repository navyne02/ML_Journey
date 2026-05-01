import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training the Computer Vision AI... 👁️")
model = SVC(gamma=0.001)
model.fit(X_train, y_train)
test_image = X_test[0] 
true_answer = y_test[0]
prediction = model.predict([test_image])
print("\n--- AI Vision Result ---")
print(f"AI Predicts this number is: {prediction[0]} 🎯")
print(f"Actual Number was: {true_answer}")
plt.imshow(test_image.reshape(8, 8), cmap=plt.cm.gray_r)
plt.title(f"AI Predicted: {prediction[0]}")
plt.axis('off') 
plt.show()

# Importing Matplotlib to draw the images, and datasets to get our sample pictures
# Importing our Support Vector Machine (the same brain we used for Spam Detection!)
# Importing train_test_split to divide our data into "study material" and "exam material"
# Loading the built-in dataset of thousands of tiny 8x8 pixel pictures of handwritten numbers
# X gets all the pixel data (the images), y gets the actual answers (the numbers 0-9)
# Shuffling and splitting the data: 80% for training (studying) and 20% for testing (the final exam)
# Creating the SVM model and tuning it specifically for image recognition (gamma=0.001)
# Training the AI using ONLY the 80% training data so it can't cheat
# Pulling the very first image out of our hidden 20% test pile to quiz the AI
# Asking the AI to look at those pixels and predict what number it sees
# --- Displaying the Results ---
# The AI actually sees a flat list of 64 numbers. We reshape it back into an 8x8 square to draw it!
# Telling Matplotlib to show the picture in black and white (gray_r)
# Hiding the graph borders and axes (Scale thevai illai!)