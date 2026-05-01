import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
print("AI is grouping the customers... 🤖")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_ 
centers = kmeans.cluster_centers_ 
print("Clustering Complete! Check the Graph 📊")
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Group Centers')
plt.title("AI Customer Segmentation (K-Means)")
plt.xlabel("Age (Scaled)")
plt.ylabel("Salary (Scaled)")
plt.legend()
plt.show()

# Importing our graphing tool, the K-Means algorithm, and a tool to generate sample data
# Creating 300 fake 'customers' who naturally fall into 3 distinct groups
# We save their data in X. Notice the '_'? We are throwing away the true answers because this is unsupervised!
# Creating the K-Means AI and telling it to hunt for exactly 3 groups
# Training the model: The AI mathematically groups the customers without us telling it who is who
# Saving the AI's final group assignments (0, 1, or 2) for every single customer
# Finding the exact center coordinate (the 'centroid') of each of the 3 groups
# --- Drawing the Graph ---
# Plotting all 300 customers. 'c=labels' tells it to color-code them based on the AI's final groups
# Plotting the 3 group centers as giant red 'X' marks so we can easily spot them
# Adding titles and labels (Age vs Salary) to make the graph look professional
# Displaying the final clustered graph!