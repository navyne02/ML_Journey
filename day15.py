from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Namma Raw Data (Age vs Salary)
# Paaka numbers romba different aana range-la irukku!
data = np.array([
    [25, 50000],
    [30, 120000],
    [22, 15000],
    [35, 90000],
    [50, 200000]
])

print("--- BEFORE SCALING (Raw Data) ---")
print(data)
print("-" * 30)

# 2. Creating the Scaler (The Magic Wand)
scaler = MinMaxScaler()

# 3. Applying the Scaling (Mathi kodu!)
scaled_data = scaler.fit_transform(data)

print("--- AFTER SCALING (0 to 1 Range) ---")
# np.round use panni 2 decimals-ku azhaga paakurom
print(np.round(scaled_data, 2))