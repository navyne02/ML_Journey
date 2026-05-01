from sklearn.preprocessing import MinMaxScaler
import numpy as np
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
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print("--- AFTER SCALING (0 to 1 Range) ---")
print(np.round(scaled_data, 2))

# Importing MinMaxScaler, a tool used to shrink large numbers down to a standard size
# Setting up raw data: Let's assume these represent [Age, Salary]
# Notice the massive difference in scale: Age is in the 10s, Salary is in the 100,000s!
# Printing the raw data to the terminal so we can compare it later
# Initializing the scaler. By default, it will compress everything into a range between 0.0 and 1.0
# The AI calculates the minimum and maximum of each column, and mathematically shrinks the data
# Printing the newly transformed data, rounding to 2 decimal places so it is easy to read