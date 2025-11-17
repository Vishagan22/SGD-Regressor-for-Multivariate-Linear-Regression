# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
# --------------------------------------------
# MULTIVARIATE LINEAR REGRESSION USING SGD
# --------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------
# SAMPLE DATA (You can replace this with your dataset)
# ----------------------------------------------------
# Features: size_of_house (sqft), number_of_occupants
# Target: price of house (in lakhs)

data = {
    "size":        [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000],
    "occupants":   [2, 3, 2, 4, 3, 5, 4, 6, 5, 7],
    "price":       [40, 55, 65, 80, 95, 110, 120, 140, 160, 180]
}

df = pd.DataFrame(data)

# Splitting features and target
X = df[["size", "occupants"]]
y = df["price"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Standardizing the data (VERY IMPORTANT for SGD)
# -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------
# SGD REGRESSOR MODEL
# -------------------------------------------------------
sgd = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01)

# Fitting the model
sgd.fit(X_train_scaled, y_train)

# Predicting
y_pred = sgd.predict(X_test_scaled)

# -------------------------------------------------------
# OUTPUT RESULTS
# -------------------------------------------------------
print("Coefficients (weights):", sgd.coef_)
print("Intercept:", sgd.intercept_)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predicting for a new house
new_data = np.array([[2100, 4]])  # 2100 sqft, 4 occupants
new_scaled = scaler.transform(new_data)
predicted_price = sgd.predict(new_scaled)

print("\nPredicted Price for 2100 sqft & 4 occupants:", predicted_price[0], "lakhs")


```

## Output:

<img width="1622" height="194" alt="image" src="https://github.com/user-attachments/assets/15ffd27b-83ea-4a5b-bee5-f63916337284" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
