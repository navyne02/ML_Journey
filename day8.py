import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Namma Data (Age vs Spending Score)
# Kavinganam: Inga namma result/answers (Pass/Fail) kudukkave illai!
data = {
    'Age': [25, 34, 22, 27, 33, 50, 60, 55, 65, 58],
    'Spending_Score': [80, 75, 90, 85, 70, 20, 15, 30, 10, 25]
}
df = pd.DataFrame(data)

# 2. Creating the ML Model (AI kitta 2 Groups-ah pirikka solrom)
model = KMeans(n_clusters=2, random_state=42)

# 3. Training (Answers illama AI-ave patterns-ah theduthu)
model.fit(df)

# 4. Result (Entha customer entha group nu paakalam)
df['Assigned_Group'] = model.labels_

print("--- AI Generated Customer Groups ---")
print(df)

# Bonus: Graph pottu paakalam!
plt.scatter(df['Age'], df['Spending_Score'], c=df['Assigned_Group'], cmap='brg', s=100)
plt.title('Customer Clustering (Unsupervised Learning)')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()