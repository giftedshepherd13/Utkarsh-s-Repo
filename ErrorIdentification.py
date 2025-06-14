import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('your_dataset.csv')  # Replace with your actual file path

# Display the first few rows of the dataset
print(df.head())

# Check the data types of each column
print(df.dtypes)

# Check for missing values in the dataset
missing_values = df.isnull().sum()

# Display columns with missing values
print(missing_values[missing_values > 0])

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Use descriptive statistics to identify potential outliers
print(df.describe())

# Visualize data to spot outliers using box plots
df.boxplot(column=['Column1', 'Column2'])  # Replace with actual column names
plt.show()

# Calculate Z-scores to identify outliers
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))

# Find rows with Z-scores greater than 3
outliers = (z_scores > 3).all(axis=1)
print(df[outliers])

# Check for unique values in categorical columns to identify inconsistencies
print(df['CategoryColumn'].unique())  # Replace with actual column name

# Use value counts to identify unusual or erroneous entries
print(df['CategoryColumn'].value_counts())

# Check numeric columns for impossible values (e.g., negative ages)
print(df[df['Age'] < 0])  # Replace “Age” with the actual column name

# Cross-validate data consistency between related columns
df['Total'] = df['Part1'] + df['Part2']  # Replace with actual column names
inconsistent_rows = df[df['Total'] != df['ExpectedTotal']]  # Replace with the actual column for the expected total
print(inconsistent_rows)

# Check for duplicate rows
duplicates = df[df.duplicated()]
print(duplicates)