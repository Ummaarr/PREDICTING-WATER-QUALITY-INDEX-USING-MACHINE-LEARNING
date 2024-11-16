import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import messagebox

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

# Model 1: MLP
def create_mlp_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

mlp_model = create_mlp_model()
mlp_model.fit(X_train_scaled, y_train, epochs=1000, batch_size=8)

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

# Function to predict WQI
def predict_wqi(input_data):
    # Convert input data to a NumPy array
    input_array = np.array(input_data, dtype=float).reshape(1, -1)

    # Scale the input data using the scaler fitted on training data
    input_scaled = scaler.transform(input_array)

    # Make the prediction using the trained MLP model
    predicted_wqi_mlp = mlp_model.predict(input_scaled)

    # Reshape input for LSTM
    input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Make the prediction using the trained LSTM model
    predicted_wqi_lstm = lstm_model.predict(input_lstm)

    return predicted_wqi_mlp[0][0], predicted_wqi_lstm[0][0]

# Create GUI
def submit():
    try:
        input_data = [
            float(entry_d.get()),
            float(entry_long.get()),
            float(entry_lat.get()),
            float(entry_ph.get()),
            float(entry_cond.get()),
            float(entry_tds.get()),
            float(entry_do.get()),
            float(entry_ca.get()),
            float(entry_mg.get()),
            float(entry_na.get()),
            float(entry_k.get()),
            float(entry_cl.get()),
            float(entry_hco3.get()),
            float(entry_so4.get()),
            float(entry_po4.get()),
            float(entry_sio2.get()),
            float(entry_18od.get()),
            float(entry_d2.get()),
            float(entry_d_ex.get()),
        ]

        predicted_mlp, predicted_lstm = predict_wqi(input_data)
        messagebox.showinfo("Predicted WQI", f"MLP Predicted WQI: {predicted_mlp:.2f}\nLSTM Predicted WQI: {predicted_lstm:.2f}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Setup tkinter window
root = tk.Tk()
root.title("Water Quality Index Prediction")
root.geometry("800x800")  # Set the window size to 800x800

# Create a title label with larger font
title_label = tk.Label(root, text="Water Quality Index Prediction", font=("Helvetica", 18, "bold"))
title_label.grid(row=0, columnspan=2, pady=10)

# Create input fields arranged in two rows
labels = ['D', 'Long', 'Lat', 'pH', 'Cond. (μS)', 'TDS(mg/l)', 'DO(mg/l)', 'Ca(mg/l)',
          'Mg(mg/l)', 'Na(mg/l)', 'K(mg/l)', 'Cl(mg/l)', 'HCO3(mg/l)', 'SO4(mg/l)',
          'PO4(mg/l)', 'SiO2', '18OD', 'D', 'd-ex']

# Create entry widgets and place them in a grid
entry_widgets = []
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i+1, column=0, padx=5, pady=5, sticky='e')  # Align labels to the right
    entry = tk.Entry(root)
    entry.grid(row=i+1, column=1, padx=5, pady=5, sticky='ew')  # Align entries to fill space
    entry_widgets.append(entry)

# Assign each entry to a variable for easy access
(entry_d, entry_long, entry_lat, entry_ph, entry_cond, entry_tds, entry_do, entry_ca,
 entry_mg, entry_na, entry_k, entry_cl, entry_hco3, entry_so4, entry_po4, entry_sio2,
 entry_18od, entry_d2, entry_d_ex) = entry_widgets

# Add weight to columns for dynamic resizing
root.grid_columnconfigure(0, weight=1)  # Label column can expand
root.grid_columnconfigure(1, weight=2)  # Entry column can expand more

# Submit button
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.grid(row=len(labels)+1, columnspan=2, pady=20)  # Place button below entries

# Run the GUI event loop
root.mainloop()