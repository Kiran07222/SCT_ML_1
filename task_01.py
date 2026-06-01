from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("data collection and understanding")
df_train = pd.read_csv("C:/Users/nani3/Desktop/house/train.csv")
df_test = pd.read_csv("C:/Users/nani3/Desktop/house/test.csv")

print("\nMissing Values in Test Data")
print(df_test.isnull().sum())
print("\nMissing Values in Training Data")
print(df_train.isnull().sum())

df_train['TotalSF'] = df_train['GrLivArea'] + df_train['TotalBsmtSF']
df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())
df_train['BedroomAbvGr'] = df_train['BedroomAbvGr'].fillna(df_train['BedroomAbvGr'].mode()[0])
df_train['TotalBath'] = (df_train['FullBath']
                        + 0.5 * df_train['HalfBath']
                        + df_train['BsmtFullBath']
                        + 0.5 * df_train['BsmtHalfBath'])

X = df_train[['TotalSF','BedroomAbvGr','TotalBath']]
y = df_train['SalePrice']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\n Model Building")
model = LinearRegression()
model.fit(X_train, y_train)


y_pred_val = model.predict(X_val)

absolute_error = mean_absolute_error(y_val, y_pred_val)


mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)

r2 = r2_score(y_val, y_pred_val)
print(f"absolute error: {absolute_error}\nMSE: {mse}\nRMSE: {rmse}\nR² Score: {r2}")
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(0)
df_test['GrLivArea'] = df_test['GrLivArea'].fillna(df_test['GrLivArea'].mean())
df_test['BedroomAbvGr'] = df_test['BedroomAbvGr'].fillna(df_test['BedroomAbvGr'].mode()[0])

df_test['FullBath'] = df_test['FullBath'].fillna(0)
df_test['HalfBath'] = df_test['HalfBath'].fillna(0)
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna(0)
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna(0)

df_test['TotalSF'] = df_test['GrLivArea'] + df_test['TotalBsmtSF']
df_test['TotalBath'] = (df_test['FullBath']
                       + 0.5 * df_test['HalfBath']
                       + df_test['BsmtFullBath']
                       + 0.5 * df_test['BsmtHalfBath'])

X_test = df_test[['TotalSF','BedroomAbvGr','TotalBath']]
X_test_scaled = scaler.transform(X_test)

predictions = model.predict(X_test_scaled)
sample_house=[[2000, 3, 2]] 
sample_house_scaled = scaler.transform(sample_house)
sample_prediction = model.predict(sample_house_scaled)
print("<------------------------------------------------------------------->\n")
print(f"predicted score is : ${sample_prediction[0]:.2f}") 
print("<------------------------------------------------------------------->\n")

save_path = r"C:\Users\nani3\Desktop"

# Graph 1: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_val)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()])
plt.savefig(f"{save_path}\\actual_vs_predicted.png")
plt.close()

# Graph 2: Error Distribution
errors = y_val - y_pred_val

plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.savefig(f"{save_path}\\error_distribution.png")
plt.close()

# Graph 3: Feature Coefficients
features = ['TotalSF', 'BedroomAbvGr', 'TotalBath']
coefficients = model.coef_

plt.figure(figsize=(8, 6))
plt.bar(features, coefficients)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Linear Regression Feature Coefficients")
plt.savefig(f"{save_path}\\feature_coefficients.png")
plt.close()

print("\nGraphs saved successfully!")