import os
import pandas as pd
import numpy as np

def expand_rows(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Repeat each unique value in the column 'YourColumnName' 30 times
    expanded_df = pd.DataFrame({
        'SpO2': np.repeat(df['SpO2'], 25)
        # Add more columns as needed
    })

    return expanded_df

def process_folders(root_folder):
    # First-level folder
    for folder_p1 in os.listdir(root_folder):
        folder_p1_path = os.path.join(root_folder, folder_p1)

        # Check if it's a folder, not a file
        if os.path.isdir(folder_p1_path):
            # Second-level folder
            for folder_vi in os.listdir(folder_p1_path):
                folder_vi_path = os.path.join(folder_p1_path, folder_vi)

                # Check if it's a folder, not a file, and contains 'v1'
                if os.path.isdir(folder_vi_path) and "v1" in folder_vi:
                    # Third-level folder
                    for folder_source in os.listdir(folder_vi_path):
                        folder_source_path = os.path.join(folder_vi_path, folder_source)

                        # Check if it's a folder, not a file, and the folder is 'source1'
                        if os.path.isdir(folder_source_path) and folder_source == 'source1':
                            # Fourth-level folder
                            for file_name in os.listdir(folder_source_path):
                                if file_name == 'gt_SpO2.csv':
                                    file_path = os.path.join(folder_source_path, file_name)
                                    print(f"Processing file: {file_path}")

                                    # Expand rows
                                    expanded_df = expand_rows(file_path)

                                    # Generate a unique output file name based on the folder structure
                                    output_file_name = f"{folder_p1}_{folder_source}_output_GT.csv"
                                    output_file_path = os.path.join('GT_output', output_file_name)

                                    # Save the expanded data to a separate CSV file
                                    expanded_df.to_csv(output_file_path, index=False)
                                    print(f"Expanded data saved to '{output_file_path}'")

# Specify the root folder path
root_folder_path = r'Z:\VIPL-HR_dataset\data'

# Execute the processing
process_folders(root_folder_path)
