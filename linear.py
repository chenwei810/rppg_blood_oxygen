import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 讀取CSV檔案
data = pd.read_csv('output_resample_filter/Sub_9_0_resample.csv')
# data = pd.read_csv('output_filter/Sub_9_0.csv')

# 處理NaN值，這裡使用平均值填補NaN值
data.fillna(data.mean(), inplace=True)

# 準備資料
X = data[['RR_filter']]  # 特徵
y = data['SPO2']  # 目標

# 切割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立線性回歸模型
model = LinearRegression()

# 訓練模型
model.fit(X_train, y_train)

# 取得模型參數
a = model.coef_[0]
b = model.intercept_

# 打印方程式
print(f"SPO2 = {a:.0f}*RR + {b:.0f}")
