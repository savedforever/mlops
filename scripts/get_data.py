# getting the housing data from scikit learn
import pandas as pd
from sklearn.datasets import fetch_california_housing
from pathlib import Path

# Fetch data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Save to a CSV file inside the 'data' directory
data_path = Path("data")
data_path.mkdir(exist_ok=True)
df.to_csv(data_path / "california_housing.csv", index=False)
print("Data saved to data/california_housing.csv")