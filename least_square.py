import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Set folder paths
a_folder_path = 'data'
b_folder_path = 'output'

# Read all CSV files in folder a_folder
# a_files = glob.glob(os.path.join(a_folder_path, '*result_Sub_1*.csv'))
a_files = glob.glob(os.path.join(a_folder_path, '*gt_SpO2*.csv'))
a_data = pd.concat([pd.read_csv(file) for file in a_files], ignore_index=True)

# Read CSV files in folder b_folder with 'a' in the filename
# b_files = glob.glob(os.path.join(b_folder_path, '*Sub_1_output*.csv'))
b_files = glob.glob(os.path.join(b_folder_path, '*video_output*.csv'))
b_data = pd.concat([pd.read_csv(file) for file in b_files], ignore_index=True)

# Print the columns of b_data
print(b_data.columns)

# Check if 'RR' exists in b_data columns
if 'RR' in b_data.columns:
    # Filter data where RR is between -10 and 10
    filtered_data = b_data[(b_data['RR'] >= -10) & (b_data['RR'] <= 10)]

    # Extract SPO2 and RR data after filtering
    x_data = filtered_data['RR']
    y_data = a_data.loc[a_data.index.isin(filtered_data.index), 'SpO2']

    # Ensure x_data and y_data have the same length
    min_length = min(len(x_data), len(y_data))
    x_data = x_data[:min_length]
    y_data = y_data[:min_length]

    # Define linear fit function
    def linear_fit(x, a, b):
        return a * x + b

    # Initial guess values
    initial_guess = [0, 1]

    # Perform linear fit
    params, covariance = curve_fit(linear_fit, x_data, y_data, p0=initial_guess)

    # Extract fit results
    a, b = params

    # Print fit results
    print(f'a: {a}, b: {b}')
    print(f'Ground-Truth_SPO2 = {a} * RR + {b}')

    # Plot the data points and fit curve
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, linear_fit(x_data, a, b), color='red', label='Linear Fit')
    plt.xlabel('RR')
    plt.ylabel('SPO2')
    plt.legend()
    plt.show()
else:
    print("'RR' column not found in b_data.")
