import pandas as pd
import numpy as np

# 1. Creating "Messy" Data (Notice the np.nan meaning empty)
data = {
    'Name': ['Naveen', 'Arun', 'Vijay', 'Karthik'],
    'Age': [22, np.nan, 24, 21],          # Arun's age is missing
    'Salary': [50000, 60000, np.nan, 45000] # Vijay's salary is missing
}

df = pd.DataFrame(data)
print("--- Messy Data (Before Cleaning) ---")
print(df)

# 2. Cleaning the Data!
# Trick 1: Fill missing salaries with 0
df['Salary'] = df['Salary'].fillna(0)

# Trick 2: Fill missing age with the "Average" age
average_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(average_age)

print("\n--- Cleaned Data (Ready for ML) ---")
print(df)