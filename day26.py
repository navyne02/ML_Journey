import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Namma Data (Fake Transactions Create Panrom)
# Normal transactions (Chinna chinna amount, normal time)
normal_transactions = np.random.randn(200, 2) * 2 + 10  

# Fraud transactions (Periya amount, odd time - Outliers)
fraud_transactions = np.random.uniform(low=-10, high=30, size=(20, 2))

# Rendu data-vaiyum onna serkurom
X = np.vstack([normal_transactions, fraud_transactions])

print(f"Total Transactions: {len(X)} (200 Normal + 20 Fraud)")

# 2. Creating the Fraud Detector (Isolation Forest)
# contamination=0.10 na "Data-la 10% fraud irukkalam" nu AI-ku hint tharom
ai_detective = IsolationForest(contamination=0.10, random_state=42)

# 3. Training and Predicting
print("AI Detective is investigating the data... 🔍")
ai_detective.fit(X)

# AI predict pannum: Normal na 1, Fraud na -1 varum
predictions = ai_detective.predict(X)

# 4. Result Analysis
anomalies_count = list(predictions).count(-1)
print(f"🚨 AI Alert: Found {anomalies_count} Suspicious Transactions!")

# 5. Visualizing the Magic
plt.figure(figsize=(10, 6))

# Normal points plot panrom
plt.scatter(X[predictions == 1][:, 0], X[predictions == 1][:, 1], 
            c='blue', label='Normal Transaction', alpha=0.6)

# Fraud points plot panrom (Red color-la danger mark maathiri)
plt.scatter(X[predictions == -1][:, 0], X[predictions == -1][:, 1], 
            c='red', marker='x', s=100, label='Fraud / Anomaly 🚨')

plt.title("Credit Card Fraud Detection (Isolation Forest) 💳")
plt.xlabel("Transaction Time Feature")
plt.ylabel("Transaction Amount Feature")
plt.legend()
plt.show()