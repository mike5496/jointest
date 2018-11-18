# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Training data
app_train = pd.read_csv('./data/application_train.csv')
print('Training data shape: ', app_train.shape)
print(app_train.head())

# Testing data features
app_test = pd.read_csv('./data/application_test.csv')
print('Testing data shape: ', app_test.shape)
print(app_test.head())

print(app_train['TARGET'].value_counts())

app_train['TARGET'].astype(int).plot.hist();

# Missing values statistics
missing_values = missing_values_table(app_train)
result = missing_values.head(20)
print(result)

# Number of each type of column
result = app_train.dtypes.value_counts()
print(result)

# Number of unique classes in each object column
result = app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
print(result)

plt.show()
