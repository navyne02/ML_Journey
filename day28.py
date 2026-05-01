import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
X = np.array([
    [12, 500, 0],  
    [2, 1500, 4],  
    [24, 600, 1],  
    [1, 2000, 5], 
    [36, 400, 0],  
    [3, 1800, 3]   
])
y = np.array([0, 1, 0, 1, 0, 1])
ai_booster = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
print("AI is learning from its mistakes... 📈")
ai_booster.fit(X, y)
new_customer = np.array([[4, 1700, 3]])
prediction = ai_booster.predict(new_customer)
print("\n--- 🏢 Customer Churn Prediction ---")
print(f"Customer Profile: {new_customer[0]}")
if prediction[0] == 1:
    print("🚨 AI Alert: This customer is going to LEAVE! Give them a discount/offer soon!")
else:
    print("✅ AI Says: This customer is SAFE and will stay with the company.")

# Importing NumPy for math, GradientBoostingClassifier for our AI, and train_test_split (though we aren't using it yet!)
# Setting up our customer data: Let's assume [Months with Company, Monthly Spend, Support Tickets]
# Notice a pattern? Customers with short tenure and high tickets seem to have issues.
# Setting up our target labels: 0 means the customer is SAFE, 1 means they will LEAVE (Churn)
# Creating the Gradient Boosting AI. 
# 'n_estimators=50' means it will build 50 small decision trees sequentially to learn the patterns
# 'learning_rate=0.1' controls how aggressively it corrects the mistakes of the previous trees
# Training the AI on our small customer database
# Creating a brand new customer profile to test: 4 months in, spends 1700, has 3 support tickets
# Asking the AI to predict if this specific customer is going to stay or churn
# --- Printing the Final Alert ---
# If the AI predicts a 1, it triggers the Churn Warning so we can send them a discount
# If it predicts a 0, the customer is happy and safe