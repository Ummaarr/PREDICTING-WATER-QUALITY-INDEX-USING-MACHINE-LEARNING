import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv('data1.csv')

# Calculate SAR
data['SAR'] = (data['Na(m g/l)'] ** 2) / np.sqrt(data['Ca(mg/ l)'] ** 2 + data['Mg(mg/l)'] ** 2)
data['PI'] = (data['Na(m g/l)'] + np.sqrt(data['HCO3(mg/l)'])) / np.sqrt(data['Ca(mg/ l)'] + data['Mg(mg/l)'] + data['Na(m g/l)']) * 100
data['Na%'] = (data['Na(m g/l)'] + data['K(mg/l)']) / (data['Ca(mg/ l)'] + data['Mg(mg/l)'] + data['Na(m g/l)'] + data['K(mg/l)']) * 100
data['RSC'] = data['HCO3(mg/l)'] - (data['Ca(mg/ l)'] + data['Mg(mg/l)'])


# Save the updated DataFrame to a new CSV file
data.to_csv('data_with_sar.csv', index=False)