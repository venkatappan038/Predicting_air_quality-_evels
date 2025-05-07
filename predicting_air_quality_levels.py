import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Generate mock air quality dataset if not already present
filename = "Mock_AirQuality.csv"

if not os.path.exists(filename):
    print("Generating mock dataset...")
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
    df_mock = pd.DataFrame(data)
    df_mock.to_csv(filename, index=False)
    print(f"Mock dataset saved as '{filename}'.")
else:
    print(f"Using existing dataset '{filename}'.")

# Step 2: Load the dataset
df = pd.read_csv(filename)

# Step 3: Feature and target separation
X = df.drop(['CO(GT)'], axis=1)  # Features
y = df['CO(GT)']                 # Target (CO levels)

# Step 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n✅ Model Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Step 8: Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual CO(GT)")
plt.ylabel("Predicted CO(GT)")
plt.title("Actual vs Predicted CO Levels (Air Quality)")
plt.grid(True)
plt.show()
