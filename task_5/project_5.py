import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Housing.csv")
print(df.info())
print(df.describe())
df.fillna(df.mean(numeric_only=True), inplace=True)
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
plt.figure()
plt.scatter(y_test, y_pred, label="Predicted Values")
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), linestyle='--', label="Regression Line")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()
n = 10
actual = y_test.values[:n]
predicted = y_pred[:n]

x = np.arange(n)
width = 0.35

plt.figure()
plt.bar(x - width/2, actual, width, label='Actual Price')
plt.bar(x + width/2, predicted, width, label='Predicted Price')

plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.title("Actual vs Predicted House Prices (Bar Chart)")
plt.legend()
plt.tight_layout()
plt.show()