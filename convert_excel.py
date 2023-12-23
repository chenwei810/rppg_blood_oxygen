import os
import pandas as pd

# Original CSV file path
input_folder = 'data'
output_folder = 'data'
target_column = 'SPO2'

# Check and create the output folder
os.makedirs(output_folder, exist_ok=True)

# List all CSV files
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Process each CSV file
for csv_file in csv_files:
    # Compose the file path
    input_file_path = os.path.join(input_folder, csv_file)
    
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Take the 12800 rows
    df = df.head(12800)

    # Perform the operation: same column, (row1+row2)/2, (row3+row4)/2, and so on
    result_df = df.groupby(df.index // 2)[target_column].mean().reset_index()

    # New CSV file path
    output_file_path = os.path.join(output_folder, f'result_{csv_file}')

    # Write the processed result to a new CSV file, retaining decimal points
    result_df.to_csv(output_file_path, index=False, float_format='%.1f')

    print(f"Already processed {input_file_path} into {output_file_path}")
