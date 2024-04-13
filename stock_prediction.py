import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Load the Data
data = pd.read_csv('stock_data.csv')

# Step 2: Preprocess the Data
data = data[['Close']]
data['Prediction'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Step 3: Split the Data into Training and Testing Sets
X = np.array(data.drop(['Prediction'], 1))
y = np.array(data['Prediction'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
predictions = model.predict(X_test)

# Step 6: Visualize the Results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
