import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'data.csv'  # Replace with your actual dataset path
data = pd.read_csv(file_path)

# Handle the 'SampleI D' column (encoding categorical data if applicable)
if 'SampleI D' in data.columns:
    # Replace with LabelEncoder if it's categorical
    label_encoder = LabelEncoder()
    data['SampleI D'] = label_encoder.fit_transform(data['SampleI D'].astype(str))

# Print original and encoded values
print("Original values and their corresponding encoded values:")
for original_value, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{original_value} -> {encoded_value}")


for col in data.select_dtypes(include=['object']).columns:

    data[col] = data[col].str.replace('−', '-', regex=False).str.strip()

# Convert all columns to numeric, forcing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Impute mean value for missing entries
data_imputed = data.fillna(data.mean(numeric_only=True))

# Check for infinite values and replace with NaN, then impute again
data_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)
data_imputed = data_imputed.fillna(data_imputed.mean())

# Exploratory Data Analysis (EDA)
# Visualizing the distribution of the target variable 'WQI'
plt.figure(figsize=(10, 5))
sns.histplot(data_imputed['WQI'], bins=30, kde=True)
plt.title('Distribution of WQI')
plt.xlabel('WQI')
plt.ylabel('Frequency')
plt.show()
# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data_imputed.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            annot_kws={"size": 8})  # Set annotation font size here
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# Features (X) and Target (y)
X = data_imputed.drop(columns=['WQI'])  # Exclude the target column
y = data_imputed['WQI']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (Neural Networks perform better with normalized data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()

# Input layer with the number of features
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Output layer for regression (no activation function)
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model and capture the training history
history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=8, validation_split=0.2)

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse:.2f}")

# Example input values
input_data = ['pre15',76.68,11.41,6.7,141,112.8,3.3,18,13.98,31,2,121.3,24.4,0.54,0.31,3.2,-4.89,-31.49,7.63]

# Convert 'pre' to a numeric label (e.g., 2 if you used factorization like before)
input_data[0] = label_encoder.transform([input_data[0]])[0]  # Convert categorical value to numeric

# Convert the input data to a NumPy array for prediction
input_array = np.array(input_data, dtype=float).reshape(1, -1)

# Apply the scaler used during training (assuming the scaler was fitted on the training data)
input_scaled = scaler.transform(input_array)

# Make the prediction using the trained neural network model
predicted_wqi = model.predict(input_scaled)

print(f"Predicted WQI: {predicted_wqi[0][0]:.2f}")