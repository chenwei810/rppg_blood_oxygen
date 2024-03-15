import os
import pandas as pd

# 定義處理資料的函數
def process_csv(input_filename):
    # Load the CSV file
    df = pd.read_csv(input_filename)
    
    # 處理資料的其餘部分，根據您的需求進行操作
    # Calculate time in seconds
    frame_rate = 30  # frames per second
    df['Time'] = df.index / frame_rate

    # Convert index to datetime
    df.index = pd.to_datetime(df['Time'], unit='s')

    # Resample R, G, B every 1 second and take the mean
    df_rgb_resampled = df[["R", "R_filter", "G", "G_filter", "B", "B_filter", "AC_red", "DC_red", "AC_green", "DC_green", "AC_blue", "DC_blue", "RR", "POS_AC", "CHROM_AC"]].resample('1s').mean().reset_index()
    # 刪除 'Time' 欄位
    df_rgb_resampled.drop(columns=['Time'], inplace=True)

    # 將處理後的資料另存為 CSV 檔案
    output_folder = 'output_resample_filter/'  # 輸出資料夾的路徑
    output_filename = os.path.join(output_folder, os.path.basename(input_filename.replace('.csv', '_resample.csv')))
    df_rgb_resampled.to_csv(output_filename, index=False)

    print(f"Processed data saved to: {output_filename}")

# 所有 CSV 檔案的清單
csv_filenames = [
        'output_filter/Sub_9_0.csv',
        'output_filter/Sub_18_0.csv',
        'output_filter/Sub_18_1.csv',
        'output_filter/Sub_19_0.csv',
        'output_filter/Sub_19_1.csv',
        'output_filter/Sub_21_1.csv',
        'output_filter/Sub_21_2.csv',
        'output_filter/Sub_22_1.csv',
        'output_filter/Sub_22_2.csv',
        'output_filter/Sub_23_1.csv',
        'output_filter/Sub_23_2.csv',
        'output_filter/Sub_24_0.csv',
        'output_filter/Sub_24_1.csv',
        'output_filter/Sub_25_0.csv',
        'output_filter/Sub_25_1.csv',
        'output_filter/Sub_26_0.csv',
        'output_filter/Sub_26_1.csv',
        'output_filter/Sub_29_0.csv',
        'output_filter/Sub_30_0.csv',
    ]

# 逐一處理所有 CSV 檔案
for filename in csv_filenames:
    process_csv(filename)
