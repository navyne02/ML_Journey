import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Namma Data (Fake Customer Data create panrom)
# 300 customers, 3 natural groups-ah irukkura maathiri data uruvakkurom
# X = [Vayasu, Sambalam]
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 2. Creating the K-Means Model
# AI kitta "Itha 3 groups-ah piri" nu solrom
print("AI is grouping the customers... 🤖")
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. Training the Model (Inga 'y' answers kidaiyathu, verum 'X' thaan!)
kmeans.fit(X)

# 4. Get the Results (AI entha point entha group nu kandupudichiduchu)
labels = kmeans.labels_ # Example: 0, 1, or 2
centers = kmeans.cluster_centers_ # Group oda center points

print("Clustering Complete! Check the Graph 📊")

# 5. Visualizing the Magic
plt.figure(figsize=(8, 6))

# Plot all points, colored by AI's prediction
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

# Plot the Centers (Centroids) in RED
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Group Centers')

plt.title("AI Customer Segmentation (K-Means)")
plt.xlabel("Age (Scaled)")
plt.ylabel("Salary (Scaled)")
plt.legend()
plt.show()