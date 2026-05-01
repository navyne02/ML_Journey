import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
digits = datasets.load_digits()
X = digits.data
y = digits.target
print(f"Original Data Shape: {X.shape} (64 Columns! 🤯)")
pca = PCA(n_components=2)
X_compressed = pca.fit_transform(X)
print(f"Compressed Data Shape: {X_compressed.shape} (Only 2 Columns! 😎)")
info_kept = sum(pca.explained_variance_ratio_) * 100
print(f"Information Kept: {info_kept:.2f}% (Just in 2 columns!)")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_compressed[:, 0], X_compressed[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.title("PCA: 64D Digits Compressed into 2D Graph 📉")
plt.xlabel("Principal Component 1 (Main Shadow)")
plt.ylabel("Principal Component 2 (Secondary Shadow)")
plt.show()

# Importing our plotting tool, the digits dataset, and PCA (our mathematical data compressor!)
# Loading the dataset of handwritten numbers
# X holds the 64 pixels for each image, y holds the true answers (0-9)
# Printing the original size of our data (64 distinct columns/dimensions!)
# Creating the PCA tool and telling it to crush our 64 dimensions down to just 2
# The magic step: The AI mathematically compresses the data into a new variable called X_compressed
# Printing the new size of our data (Only 2 columns left!)
# Calculating exactly how much of the original "meaning" survived the extreme compression
# Printing the percentage of information kept (usually around 28% for 2 components here)
# --- Drawing the Graph ---
# Creating a large canvas for our plot
# Plotting the newly compressed 2D data! We color-code each dot by its true number class (c=y)
# Adding a color legend to the side so we know which color represents which drawn digit
# Adding a title to our graph
# Labeling the X and Y axes as our new "Principal Components" (the mathematical shadows of the data)
# Showing the final 2D visualization of our 64D data!