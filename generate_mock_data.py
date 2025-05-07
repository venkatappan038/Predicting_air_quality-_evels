import pandas as pd
import numpy as np

# Create a mock dataset with 100 samples and air quality features
data = {
    'CO(GT)': np.random.rand(100) * 10,           # Carbon Monoxide (mg/m³)
    'PT08.S1(CO)': np.random.rand(100) * 100,     # Sensor 1
    'PT08.S2(NMHC)': np.random.rand(100) * 50,    # Sensor 2
    'PT08.S3(NOx)': np.random.rand(100) * 60,     # Sensor 3
    'PT08.S4(NO2)': np.random.rand(100) * 80,     # Sensor 4
    'PT08.S5(O3)': np.random.rand(100) * 90,      # Sensor 5
    'T': np.random.rand(100) * 40,                # Temperature (°C)
    'RH': np.random.rand(100) * 100,              # Relative Humidity (%)
    'AH': np.random.rand(100) * 1                 # Absolute Humidity
}

# Create the DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("Mock_AirQuality.csv", index=False)

print("✅ Mock dataset 'Mock_AirQuality.csv' created successfully.")
