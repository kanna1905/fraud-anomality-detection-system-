import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the data
# Replace 'transactions.csv' with the path to your dataset
data = pd.read_csv('transactions.csv')
print(data.columns)
