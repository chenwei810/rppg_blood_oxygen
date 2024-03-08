import os
import pandas as pd

def copy_spo2_column(source_filename, destination_filename):
    # 讀取源檔案和目標檔案
    source_df = pd.read_csv(source_filename)
    
    # 如果目標檔案不存在，則創建一個新的 DataFrame 並新增 'SPO2' 欄位
    if not os.path.exists(destination_filename):
        destination_df = pd.DataFrame()
        destination_df['SPO2'] = source_df['SPO2']
    else:
        destination_df = pd.read_csv(destination_filename)

    # 將 SPO2 欄位從源 DataFrame 複製到目標 DataFrame
    destination_df['SPO2'] = source_df['SPO2']

    # 將目標 DataFrame 另存為 CSV 檔案
    destination_df.to_csv(destination_filename, index=False)
    print(f"SPO2 column copied from '{source_filename}' to '{destination_filename}'.")

# 指定源檔案和目標檔案的列表
source = [
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_9_0\BESTSPO2_Sub_9_0_2024-03-04 19_35_10.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_18_0\BESTSPO2_Sub_18_0_2024-03-01 13_21_30.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_18_1\BESTSPO2_Sub_18_1_2024-03-01 13_27_50.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_19_0\BESTSPO2_Sub_19_0_2024-03-01 14_43_30.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_19_1\BESTSPO2_Sub_19_1_2024-03-01 14_49_20.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_21_1\BESTSPO2_Sub_21_1_2024-02-29 11_44_40.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_21_2\BESTSPO2_Sub_21_2_2024-02-29 11_51_30.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_22_1\BESTSPO2_Sub_22_1_2024-02-29 14_45_20.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_22_2\BESTSPO2_Sub_22_2_2024-02-29 14_51_50.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_23_1\BESTSPO2_Sub_23_1_2024-02-29 15_39_10.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_23_2\BESTSPO2_Sub_23_2_2024-02-29 15_50_40.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_24_0\BESTSPO2_Sub_24_0_2024-03-01 11_27_20.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_24_1\BESTSPO2_Sub_24_1_2024-03-01 11_35_30.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_25_0\BESTSPO2_Sub_25_0_2024-03-01 15_23_40.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_25_1\BESTSPO2_Sub_25_1_2024-03-01 15_29_50.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_26_0\BESTSPO2_Sub_26_0_2024-03-04 14_49_40.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_26_1\BESTSPO2_Sub_26_1_2024-03-04 14_55_40.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_29_0\BESTSPO2_Sub_29_0_2024-03-05 17_59_00.csv',
    'Z:\SPO2_IRB_Lab_dataset\EE\Sub_30_0\BESTSPO2_Sub_30_0_2024-03-05 19_23_50.csv',
]

destination = [
    'output_resample/Sub_9_0_resample.csv',
    'output_resample/Sub_18_0_resample.csv',
    'output_resample/Sub_18_1_resample.csv',
    'output_resample/Sub_19_0_resample.csv',
    'output_resample/Sub_19_1_resample.csv',
    'output_resample/Sub_21_1_resample.csv',
    'output_resample/Sub_21_2_resample.csv',
    'output_resample/Sub_22_1_resample.csv',
    'output_resample/Sub_22_2_resample.csv',
    'output_resample/Sub_23_1_resample.csv',
    'output_resample/Sub_23_2_resample.csv',
    'output_resample/Sub_24_0_resample.csv',
    'output_resample/Sub_24_1_resample.csv',
    'output_resample/Sub_25_0_resample.csv',
    'output_resample/Sub_25_1_resample.csv',
    'output_resample/Sub_26_0_resample.csv',
    'output_resample/Sub_26_1_resample.csv',
    'output_resample/Sub_29_0_resample.csv',
    'output_resample/Sub_30_0_resample.csv',
]
# 逐一處理每對檔案
for source_file, destination_file in zip(source, destination):
    copy_spo2_column(source_file, destination_file)
