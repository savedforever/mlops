# train.py to train models Linear Regression and Decision Tree on housing data
import mlflow 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Set the experiment name
mlflow.set_experiment("California Housing Prediction")

# Load data
df = pd.read_csv("data/california_housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# 1. Train Linear Regression
with mlflow.start_run(run_name="Linear Regression"):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    mlflow.log_param("model_type", "LinearRegression")   
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(lr, "model", input_example=X_train.head(1))
    print(f"Linear Regression RMSE: {rmse}")

# 2. Train Decision Tree Regressor
with mlflow.start_run(run_name="Decision Tree"):
    # Define parameters to try
    max_depth = 5
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=20)
    dt.fit(X_train, y_train)
    predictions = dt.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    mlflow.log_param("model_type", "DecisionTree")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(dt, "model", input_example=X_train.head(1))
    print(f"Decision Tree RMSE: {rmse}")