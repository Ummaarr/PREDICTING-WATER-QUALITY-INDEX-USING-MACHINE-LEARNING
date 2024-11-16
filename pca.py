import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import bartlett
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer  # For Varimax rotation
import seaborn as sns
# Load the dataset
data = pd.read_csv('data.csv')  # Replace with your actual CSV file path

# Inspecting the dataset
print(data.head())

# Drop non-numeric columns (SampleID, Lat, Long)
data_numeric = data.drop(['SampleI D', 'Long.', 'Lat.'], axis=1)

# Handle missing values (if any)
data_numeric.fillna(data_numeric.mean(), inplace=True)

# Separating the features and the target variable (WQI in this case)
X = data_numeric.drop('WQI', axis=1)
y = data_numeric['WQI']

# Inspecting the dataset
print(data.head())

# Drop non-numeric columns (SampleID, Lat, Long)
data_numeric = data.drop(['SampleI D', 'Long.', 'Lat.'], axis=1)

# Handle missing values (if any)
data_numeric.fillna(data_numeric.mean(), inplace=True)

# Calculate the correlation coefficient matrix
correlation_matrix = data_numeric.corr()

# Display the correlation coefficient matrix
print("Correlation Coefficient Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Coefficient Matrix')
plt.show()
# Standardizing the data (mean = 0 and variance = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Conduct Bartlett's test for sphericity
statistic, p_value = bartlett(*[X_scaled[:, i] for i in range(X_scaled.shape[1])])
print(f"Bartlett's test statistic: {statistic}, p-value: {p_value}")

# Apply PCA
pca = PCA(n_components=6)  # Choose the number of components you want to keep
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Display explained variance for each component
print("Explained Variance by Each Principal Component:")
for i, ev in enumerate(explained_variance):
    print(f"PC{i+1}: {ev:.4f}")

print("Cumulative Explained Variance:")
for i, cv in enumerate(cumulative_variance):
    print(f"PC{i+1}: {cv:.4f}")

# Visualizing the explained variance ratio
plt.figure(figsize=(8,6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, align='center', label='Individual Explained Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.show()

# Perform Varimax rotation
fa = FactorAnalyzer(n_factors=6, rotation='varimax')
fa.fit(X_scaled)

# Get factor loadings after rotation
loadings = fa.loadings_

# Display the PCA loadings (component contributions)
pca_loadings = pd.DataFrame(loadings, index=X.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
print("PCA Loadings (Component Contributions):")
print(pca_loadings)

# Visualizing the PCA components on a 2D plot (if you want to look at first two components)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - First Two Components')
plt.colorbar(label='WQI')
plt.show()