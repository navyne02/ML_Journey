import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = {
    'Age': [25, 34, 22, 27, 33, 50, 60, 55, 65, 58],
    'Spending_Score': [80, 75, 90, 85, 70, 20, 15, 30, 10, 25]
}
df = pd.DataFrame(data)
model = KMeans(n_clusters=2, random_state=42)
model.fit(df)
df['Assigned_Group'] = model.labels_
print("--- AI Generated Customer Groups ---")
print(df)
plt.scatter(df['Age'], df['Spending_Score'], c=df['Assigned_Group'], cmap='brg', s=100)
plt.title('Customer Clustering (Unsupervised Learning)')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()

# Creating a dataset containing customer ages and their corresponding spending scores
# Converting the raw data into a structured Pandas DataFrame (df)
# Setting up the KMeans model to find 2 distinct customer groups (clusters)
# Training the model so it can discover hidden patterns and group the data automatically
# Assigning the AI-generated group label back to each customer in our table
# Plotting the customers on a scatter graph, color-coding them based on their assigned group