import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("sales.csv")

# Prophet format
df.columns = ["ds", "y"]

# Model
model = Prophet()
model.fit(df)

# Future prediction
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
model.plot(forecast)
plt.title("Sales Forecast")
plt.show()