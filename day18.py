from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. Namma Raw Data (Age vs Salary)
X = np.array([
    [22, 15000],  # Junior
    [25, 40000],  # Junior
    [35, 90000],  # Senior
    [50, 150000]  # Senior
])
# 0 = Junior, 1 = Senior
y = np.array([0, 0, 1, 1]) 

# 2. Creating the Pipeline (The Assembly Line)
# Step 1: Scale the data (0 to 1)
# Step 2: Use Logistic Regression to classify
my_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('ai_model', LogisticRegression())
])

# 3. Train the entire pipeline at once!
print("Starting the Assembly Line... ⚙️")
my_pipeline.fit(X, y)

# 4. Test with a New Person
# Vayasu 30, Sambalam 60,000 -> Ivar Junior ah? Senior ah?
new_person = np.array([[30, 60000]])

# Kavinganam: Namma manually scale panna thevai illai, Pipeline athave paathukkum!
prediction = my_pipeline.predict(new_person)

print("\n--- Pipeline Result ---")
if prediction[0] == 1:
    print("AI predicts: This person is a SENIOR! 👔")
else:
    print("AI predicts: This person is a JUNIOR! 💼")