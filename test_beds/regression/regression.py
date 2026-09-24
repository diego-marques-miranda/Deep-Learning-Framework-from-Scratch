from pathlib import Path

import numpy as np
import pickle
import pandas as pd

import layers
import activation_functions
import loss_functions
import optimizers
from data.scaler import MinMaxScaler 
from data.dataloader import DataLoader
from metrics.metrics import MAPE, R2
from data.utils import train_test_split
from model import Model 
from data.preprocessing import MedianImputer

# Initialize controlled random number generator for reproducibility
rng = np.random.default_rng(112)

# Load dataset & extract target/features
DATA_PATH = Path(__file__).parent / "housing.csv"

data = pd.read_csv(DATA_PATH)

y = data['median_house_value'].values.reshape(-1, 1)
X = data[['total_rooms', 
          'housing_median_age', 
          'latitude', 
          'longitude', 
          'total_bedrooms', 
          'population', 
          'households', 
          'median_income'
          ]].to_numpy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Impute missing feature values using training set medians to prevent leakage
medianImputer = MedianImputer()
medianImputer.fit(X_train)
X_train = medianImputer.transform(X_train)
X_test = medianImputer.transform(X_test)

# Fit scalers on train set only to prevent data leakage, then scale features and target
model = Model()
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(X_train)
scaler_y.fit(y_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

# Setup batch iterators with deterministic shuffling
train_loader = DataLoader(X_train, y_train, batch_size=32, rng=rng)
test_loader = DataLoader(X_test, y_test, batch_size=32, rng=rng)

# Build architecture
model.add(layers.Layer_Dense(8, 32, activation_func=activation_functions.GELU(), rng=rng))
model.add(layers.Layer_Dense(32, 32, activation_func=activation_functions.GELU(), rng=rng))
model.add(layers.Layer_Dense(32, 1, activation_func=activation_functions.Linear(), rng=rng))

# Compile with loss and optimizer
model.compile(
    loss=loss_functions.Mse(),
    optimizer=optimizers.Adam(learning_rate=0.0001)
)

print("Training model...")

train_metrics = [R2()]
test_metrics = [R2()]

# Run training loop with validation
model.fit(
    train_dataloader=train_loader, 
    val_dataloader=test_loader, 
    epochs=500, 
    train_metrics=train_metrics, 
    val_metrics=test_metrics
) 

print("\nSaving model weights...")
model.save('house_pricing_weights.pkl')