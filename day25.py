import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# 1. Load the Data (Kaiyala ezhuthuna numbers)
# Intha data-la 64 pixels (columns) irukku
digits = datasets.load_digits()
X = digits.data
y = digits.target

print(f"Original Data Shape: {X.shape} (64 Columns! 🤯)")

# 2. Apply PCA (The Magic Compressor)
# Namma AI kitta "Itha verum 2 columns-ah mathi kudu" nu solrom
pca = PCA(n_components=2)

# 3. Compress the Data
# 64 dimension la irunthu 2 dimension ku data kuraiyuthu
X_compressed = pca.fit_transform(X)

print(f"Compressed Data Shape: {X_compressed.shape} (Only 2 Columns! 😎)")

# 4. Check how much information we kept
# (Explained variance ratio tells us the percentage of info retained)
info_kept = sum(pca.explained_variance_ratio_) * 100
print(f"Information Kept: {info_kept:.2f}% (Just in 2 columns!)")

# 5. Visualize the 64-column data in a 2D Graph!
plt.figure(figsize=(10, 8))
# X_compressed[:, 0] is the 1st column, X_compressed[:, 1] is the 2nd column
scatter = plt.scatter(X_compressed[:, 0], X_compressed[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)

# Add a color bar to show which color is which number (0-9)
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.title("PCA: 64D Digits Compressed into 2D Graph 📉")
plt.xlabel("Principal Component 1 (Main Shadow)")
plt.ylabel("Principal Component 2 (Secondary Shadow)")
plt.show()