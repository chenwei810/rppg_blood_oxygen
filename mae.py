import pandas as pd
from sklearn.metrics import mean_absolute_error

# 讀取CSV文件
data = pd.read_csv("output_resample_filter/Sub_9_0_resample.csv")

# 提取SPO2和predict_SPO2列
SPO2 = data['SPO2']
Pridict_SPO2 = data['Pridict_SPO2']
# 將NaN值替換為中位數
SPO2.fillna(SPO2.median(), inplace=True)
Pridict_SPO2.fillna(Pridict_SPO2.median(), inplace=True)

# 計算MAE
mae = mean_absolute_error(SPO2, Pridict_SPO2)
# 格式化MAE顯示為小數點第二位
mae_formatted = "{:.2f}".format(mae)

print("Mean Absolute Error (MAE):", mae_formatted)
