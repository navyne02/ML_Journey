import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
normal_transactions = np.random.randn(200, 2) * 2 + 10  
fraud_transactions = np.random.uniform(low=-10, high=30, size=(20, 2))
X = np.vstack([normal_transactions, fraud_transactions])
print(f"Total Transactions: {len(X)} (200 Normal + 20 Fraud)")
ai_detective = IsolationForest(contamination=0.10, random_state=42)
print("AI Detective is investigating the data... 🔍")
ai_detective.fit(X)
predictions = ai_detective.predict(X)
anomalies_count = list(predictions).count(-1)
print(f"🚨 AI Alert: Found {anomalies_count} Suspicious Transactions!")
plt.figure(figsize=(10, 6))
plt.scatter(X[predictions == 1][:, 0], X[predictions == 1][:, 1], 
            c='blue', label='Normal Transaction', alpha=0.6)
plt.scatter(X[predictions == -1][:, 0], X[predictions == -1][:, 1], 
            c='red', marker='x', s=100, label='Fraud / Anomaly 🚨')
plt.title("Credit Card Fraud Detection (Isolation Forest) 💳")
plt.xlabel("Transaction Time Feature")
plt.ylabel("Transaction Amount Feature")
plt.legend()
plt.show()

# Importing NumPy for math, Matplotlib for graphing, and IsolationForest for finding outliers
# Creating 200 'normal' transactions that are tightly clustered together in behavior
# Creating 20 fake 'fraud' transactions that are randomly scattered all over the place
# Stacking them together into one single dataset (X) of 220 transactions
# Printing the total count so we know what the AI is dealing with
# Setting up the AI Detective. 'contamination=0.10' tells it to expect about 10% of the data to be suspicious
# Training the AI. It randomly slices the data to see which points are easiest to separate
# Asking the AI to grade the data. It returns '1' for Normal and '-1' for Fraud (Anomalies)
# Counting exactly how many '-1's the AI flagged so we can sound the alarm
# --- Drawing the Graph ---
# Creating a large canvas for our visualization
# Plotting all the transactions the AI labeled as '1' (Normal) as soft blue dots
# Plotting all the transactions the AI labeled as '-1' (Fraud) as giant red 'X' marks
# Adding our titles and axis labels
# Showing the final security report graph!