import os
import pandas as pd

# 指定原始資料夾的路徑
gt_folder_path = 'GT_output'
video_folder_path = 'video_output'

# 指定新資料夾的路徑
output_folder_path = 'merged_output'

# 取得所有 GT_output 資料夾中的 CSV 檔案
gt_csv_files = [f for f in os.listdir(gt_folder_path) if f.endswith('.csv')]

# 取得所有 video_output 資料夾中的 CSV 檔案
video_csv_files = [f for f in os.listdir(video_folder_path) if f.endswith('.csv')]

# 建立新的資料夾
os.makedirs(output_folder_path, exist_ok=True)

# 找出名稱相同的檔案進行合併
for gt_csv_file in gt_csv_files:
    # 檢查檔案名稱是否存在於 video_output 資料夾中
    if gt_csv_file in video_csv_files:
        # 構建完整路徑
        gt_csv_path = os.path.join(gt_folder_path, gt_csv_file)
        video_csv_path = os.path.join(video_folder_path, gt_csv_file)
        
        # 讀取 CSV 檔案
        gt_data = pd.read_csv(gt_csv_path)
        video_data = pd.read_csv(video_csv_path)
        
        # 合併資料
        merged_data = pd.concat([gt_data, video_data], axis=1)
        
        # 構建新的檔案路徑
        new_csv_path = os.path.join(output_folder_path, f'merged_{gt_csv_file}')
        
        # 寫入合併後的 CSV 檔案到新的資料夾
        merged_data.to_csv(new_csv_path, index=False)

print("合併完成。檔案存儲在新的資料夾:", output_folder_path)
