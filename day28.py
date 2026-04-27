import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# 1. Namma Data (Customer Churn Data)
# Columns: [Months Subscribed, Monthly Bill (Rs), Support Tickets Raised]
X = np.array([
    [12, 500, 0],  # Happy customer (Long time, low bill, no complaints)
    [2, 1500, 4],  # Unhappy (New, high bill, 4 complaints) -> Will leave!
    [24, 600, 1],  # Happyy
    [1, 2000, 5],  # Very Unhappy
    [36, 400, 0],  # Loyal customer
    [3, 1800, 3]   # Unhappy
])

# 0 = Stay (Company la iruppanga), 1 = Churn (Company ah vittu poiduvanga)
y = np.array([0, 1, 0, 1, 0, 1])

# 2. Creating the Gradient Boosting Model (The Kaggle King!)
# n_estimators=50 na 50 stages of learning from mistakes!
# learning_rate=0.1 na evlo fast-ah kathukkanum nu artham
ai_booster = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)

# 3. Training the AI
print("AI is learning from its mistakes... 📈")
ai_booster.fit(X, y)

# 4. Test with a New Customer
# 4 Months aachu, Bill Rs.1700, 3 Complaints kuduthurukanga
new_customer = np.array([[4, 1700, 3]])

prediction = ai_booster.predict(new_customer)

print("\n--- 🏢 Customer Churn Prediction ---")
print(f"Customer Profile: {new_customer[0]}")

if prediction[0] == 1:
    print("🚨 AI Alert: This customer is going to LEAVE! Give them a discount/offer soon!")
else:
    print("✅ AI Says: This customer is SAFE and will stay with the company.")
