import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# 讀取 CSV 檔案
file_path = r"Z:\SPO2_IRB_Lab_dataset\EE\Sub_22_2\BESTPPGandEKGandIR_Sub_22_2_2024-02-29 14_51_50.csv"  # 替換成你的 CSV 檔案路徑
data = pd.read_csv(file_path)

# 設置參數
order = 79 # 濾波器階數
nyquist = 0.5 * 30  # Nyquist 頻率，這裡假設取樣頻率為 400 Hz
lowcut = 0.67
highcut = 1.67
cutoff = [lowcut / nyquist, highcut / nyquist]  # 截至頻率，轉換為正規化頻率

# 計算 FIR 濾波器係數
coefficients = firwin(order, cutoff, pass_zero=False)

# 將數據轉換為 numpy 陣列
PPG_org = np.array(data['PPG_org'])
IR_org = np.array(data['ir_org'])

# 應用 FIR 濾波器到 PR_normalized 資料
PPG_org_filtered = lfilter(coefficients, 1.0, PPG_org)
IR_org_filtered = lfilter(coefficients, 1.0, IR_org)

# 將處理完的結果添加到數據中
data['PPG_org_filtered'] = PPG_org_filtered
data['ir_org_filtered'] = IR_org_filtered

# 截取第150帧之后的数据
data = data.iloc[150:]

# 儲存處理後的資料到新的 CSV 檔案
output_file_path = 'filtered_data.csv'  # 替換成你要儲存的檔案路徑
data.to_csv(output_file_path, index=False)

print(f"Filtered data saved to '{output_file_path}'")

# # 視覺化濾波後的結果
# plt.plot(data['PPG_org'], label='PPG_org')
# plt.plot(data['ir_org'], label='ir_org')

# plt.plot(data['PPG_org_filtered'], label='PPG_org_filtered')
# plt.plot(data['ir_org_filtered'], label='ir_org_filtered')
# # plt.plot(data['SPO2'], label='SPO2')

# plt.xlabel('Time Count')
# plt.ylabel('Color Value')

# plt.title('PPG vs IR')
# plt.legend()
# plt.show()
