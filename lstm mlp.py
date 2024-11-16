import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'data.csv'  # Replace with your actual dataset path
data = pd.read_csv(file_path)

# Handle the 'SampleI D' column (encoding categorical data if applicable)
if 'SampleI D' in data.columns:
    label_encoder = LabelEncoder()
    data['SampleI D'] = label_encoder.fit_transform(data['SampleI D'].astype(str))

# Clean numeric columns: replace non-standard hyphens and strip whitespace
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.replace('−', '-', regex=False).str.strip()

# Convert all columns to numeric, forcing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Impute mean value for missing entries
data_imputed = data.fillna(data.mean(numeric_only=True))

# Check for infinite values and replace with NaN, then impute again
data_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)
data_imputed = data_imputed.fillna(data_imputed.mean())

# Features (X) and Target (y)
X = data_imputed.drop(columns=['WQI'])  # Exclude the target column
y = data_imputed['WQI']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (Neural Networks perform better with normalized data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to create and evaluate a model
def evaluate_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=8):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return mse, history

# Model 1: MLP with Hyperparameter Tuning
param_grid_mlp = {
    'units': [64, 128],
    'dropout': [0.2, 0.3],
}

def create_mlp_model(units, dropout):
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

mlp_model = create_mlp_model(64, 0.2)
mlp_model.fit(X_train_scaled, y_train, epochs=1000, batch_size=8)

# Evaluate MLP Model
mlp_mse, mlp_history = evaluate_model(mlp_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Model 2: LSTM with Reshaping for LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

lstm_model = create_lstm_model()
lstm_model.fit(X_train_lstm, y_train, epochs=1000, batch_size=8)

# Evaluate LSTM Model
lstm_mse, lstm_history = evaluate_model(lstm_model, X_train_lstm, y_train, X_test_lstm, y_test)

# Plot training loss for MLP
plt.figure(figsize=(10, 5))
plt.plot(mlp_history.history['loss'], label='MLP Train Loss')
plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')
plt.title('MLP Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training loss for LSTM
plt.figure(figsize=(10, 5))
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.title('LSTM Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on test data using MLP
mlp_y_pred = mlp_model.predict(X_test_scaled)



# Example input values
input_data = ['pre15',76.69,11.4,6.8,259,207.2,3,28,6.66,37,4,85.2,97.6,0.06,0.04,7.13,-4.83,-32.19,6.45]

# Convert 'pre' to a numeric label
input_data[0] = label_encoder.transform([input_data[0]])[0]

# Convert the input data to a NumPy array for prediction
input_array = np.array(input_data, dtype=float).reshape(1, -1)

# Scale the input data using the scaler fitted on training data
input_scaled = scaler.transform(input_array)

# Make the prediction using the trained MLP model
predicted_wqi_mlp = mlp_model.predict(input_scaled)
print(f"Predicted WQI (MLP): {predicted_wqi_mlp[0][0]:.2f}")

# Reshape input for LSTM (needs to be 3D: samples, timesteps, features)
input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

# Make the prediction using the trained LSTM model
predicted_wqi_lstm = lstm_model.predict(input_lstm)
print(f"Predicted WQI (LSTM): {predicted_wqi_lstm[0][0]:.2f}")