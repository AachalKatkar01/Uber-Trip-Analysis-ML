import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Uber-Jan-Feb-FOIL (3).csv")

# Convert date column (lowercase)
df['date'] = pd.to_datetime(df['date'])

# Feature Engineering
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day
df['Weekday'] = df['date'].dt.weekday

# Visualization
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['trips'])
plt.title("Trips Over Time")
plt.xticks(rotation=45)
plt.show()

# Machine Learning
X = df[['active_vehicles', 'Month', 'Day', 'Weekday']]
y = df['trips']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))