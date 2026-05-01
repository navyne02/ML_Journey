from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
X = np.array([
    [22, 15000], 
    [25, 40000],  
    [35, 90000],  
    [50, 150000]  
])
y = np.array([0, 0, 1, 1]) 
my_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('ai_model', LogisticRegression())
])
print("Starting the Assembly Line... ⚙️")
my_pipeline.fit(X, y)
new_person = np.array([[30, 60000]])
prediction = my_pipeline.predict(new_person)
print("\n--- Pipeline Result ---")
if prediction[0] == 1:
    print("AI predicts: This person is a SENIOR! 👔")
else:
    print("AI predicts: This person is a JUNIOR! 💼")
    
# Importing Pipeline, a tool that chains multiple data processing steps together
# Importing our Scaler (from Day 15) and our Logistic Regression model (from Day 7)
# Setting up our raw data: Let's assume this is [Age, Salary]
# The final results: 0 means JUNIOR, 1 means SENIOR
# Building the Assembly Line (Pipeline): 
# Step 1: The 'scaler' automatically squishes the raw numbers down to a 0-to-1 range
# Step 2: The 'ai_model' takes that perfectly scaled data and learns the patterns
# Training the Pipeline: This single command scales the data AND trains the model all at once!
# Setting up a brand new person: Age 30, Salary 60,000
# Asking the pipeline to predict. It will automatically scale this new data before predicting!
# Printing the final prediction to see if they are a Junior or Senior