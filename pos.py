import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.signal import firwin

def mask_nose_forehead(img, face_landmarks):
    # 定義額頭區域的重要點索引
    bigger_forehead_indices = [109, 10, 338, 337, 151, 108]
    # bigger_forehead_indices = [128, 357, 188, 412, 196, 419, 197, 114, 217, 236, 3, 195, 248, 456, 437, 343] #鼻子人中
    # 創建一個與原始圖像相同大小的零矩陣，用於創建額頭區域的遮罩
    mask_forehead = np.zeros_like(img)
    # 提取額頭區域的坐標點，根據臉部關鍵點的坐標縮放
    pts_forehead = np.array([(int(face_landmarks.landmark[i].x * img.shape[1]), 
                              int(face_landmarks.landmark[i].y * img.shape[0])) for i in bigger_forehead_indices], np.int32)
    # 將坐標點轉換為遮罩中的多邊形
    pts_forehead = pts_forehead.reshape((-1, 1, 2))
    # 使用cv2.fillPoly填充多邊形區域，顏色為白色 (255, 255, 255)
    cv2.fillPoly(mask_forehead, [pts_forehead], (255, 255, 255))
    # 使用位元運算計算額頭區域的遮罩後的圖像
    forehead_masked = cv2.bitwise_and(img, mask_forehead)
    # 找到被遮罩部分的非黑色區域，並計算通道的平均值
    forehead_masked_non_black = forehead_masked[np.all(forehead_masked != [0, 0, 0], axis=-1)]
    forehead_masked_non_black_flat = forehead_masked_non_black.reshape(-1, 3)
    r_mean = np.mean(forehead_masked_non_black_flat[:, 2])
    g_mean = np.mean(forehead_masked_non_black_flat[:, 1])
    b_mean = np.mean(forehead_masked_non_black_flat[:, 0])
    # 顯示遮罩後的額頭區域和原始圖像
    cv2.imshow('forehead', forehead_masked)
    cv2.imshow('img', img)
    # 返回計算得到的顏色平均值
    return {"R": r_mean, "G": g_mean, "B": b_mean}

def process_video(cap, face_mesh, csv_filename):
    df = pd.DataFrame(columns=["R", "G", "B", "DC_red", "DC_green", "DC_blue", "AC_red", "AC_green", "AC_blue"])

    buffer_size = 30
    red_buffer, green_buffer, blue_buffer = [], [], []
    AC_red_buffer, AC_green_buffer, AC_blue_buffer = [], [], []

    POS_s1_buffer, POS_s2_buffer = [], []
    pr_raw_values = []

    CHROM_x_buffer, CHROM_y_buffer = [], []
    CHROM_raw_values = []

    frame_count = 0

    frame = 1
    while True:
        ret, img_rgb = cap.read()
        if not ret:
            break
        frame_count += 1

        img_rgb2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb2)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                forehead_data = mask_nose_forehead(img_rgb, face_landmarks)

                df = pd.concat([df, pd.DataFrame({
                    "R": [forehead_data["R"]],
                    "G": [forehead_data["G"]],
                    "B": [forehead_data["B"]]
                })], ignore_index=True)

                # POS Algorithm-----------------------------------------
                POS_matrix_operation_result = np.array([
                    [0, 1, -1],
                    [-2, 1, 1]
                ]).dot([forehead_data["R"], forehead_data["G"], forehead_data["B"]])
                
                POS_s1_buffer.append(POS_matrix_operation_result[0])
                POS_s2_buffer.append(POS_matrix_operation_result[1])

                if len(POS_s1_buffer) > buffer_size:
                    POS_s1_buffer.pop(0)
                    POS_s2_buffer.pop(0)

                # POS Algorithm END-----------------------------------------

                # CHROM Algorithm---------------------------------------
                CHROM_matrix_operation_result = np.array([
                    [3, -2, 0],
                    [1.5, 1, -1.5]
                ]).dot([forehead_data["R"], forehead_data["G"], forehead_data["B"]])
                CHROM_x_buffer.append(CHROM_matrix_operation_result[0])
                CHROM_y_buffer.append(CHROM_matrix_operation_result[1])

                if len(CHROM_x_buffer) > buffer_size:
                    CHROM_x_buffer.pop(0)
                    CHROM_y_buffer.pop(0)

                # CHROM Algorithm END---------------------------------------

                red_buffer.append(forehead_data["R"])
                green_buffer.append(forehead_data["G"])
                blue_buffer.append(forehead_data["B"])

                if len(red_buffer) > buffer_size:
                    red_buffer.pop(0)
                    green_buffer.pop(0)
                    blue_buffer.pop(0)

                if frame > buffer_size:

                    # For SPO2 calculation
                    DC_red = np.mean(red_buffer)
                    DC_green = np.mean(green_buffer)
                    DC_blue = np.mean(blue_buffer)

                    AC_red = forehead_data["R"] - DC_red
                    AC_green = forehead_data["G"] - DC_green
                    AC_blue = forehead_data["B"] - DC_blue

                    AC_red_buffer = np.append(AC_red_buffer, AC_red)
                    AC_green_buffer = np.append(AC_green_buffer, AC_green)
                    AC_blue_buffer = np.append(AC_blue_buffer, AC_blue)

                    if len(AC_red_buffer) > 30:
                        AC_red_buffer = AC_red_buffer[1:]
                        AC_green_buffer = AC_green_buffer[1:]
                        AC_blue_buffer = AC_blue_buffer[1:]

                    max_AC_red = np.max(AC_red_buffer)
                    max_AC_green = np.max(AC_green_buffer)
                    max_AC_blue = np.max(AC_blue_buffer)
                    min_AC_red = np.min(AC_red_buffer)
                    min_AC_green = np.min(AC_green_buffer)
                    min_AC_blue = np.min(AC_blue_buffer)

                    # Check if AC_red and AC_blue are positive
                    
                    RR = np.log((max_AC_red-min_AC_red)) / np.log((max_AC_blue-min_AC_blue))
                
                
                    
                    df.loc[df.index[-1], 'DC_red'] = DC_red
                    df.loc[df.index[-1], 'DC_green'] = DC_green
                    df.loc[df.index[-1], 'DC_blue'] = DC_blue
                    df.loc[df.index[-1], 'AC_red'] = max_AC_red - min_AC_red
                    df.loc[df.index[-1], 'AC_green'] = max_AC_green - min_AC_green
                    df.loc[df.index[-1], 'AC_blue'] = max_AC_blue - min_AC_blue
                    df.loc[df.index[-1], 'RR'] = RR

                    # For POS Algorithm-----------------------------------------
                    std_s1 = np.std(POS_s1_buffer)
                    std_s2 = np.std(POS_s2_buffer)
                    alpha = std_s1 / std_s2
                    if len(pr_raw_values) > buffer_size:
                        pr_raw_values.pop(0)

                    POS_raw = std_s1 + (alpha * std_s2)
                    pr_raw_values.append(POS_raw)# 計算 PR_raw 平均值
                    POS_mean = np.mean(pr_raw_values)# 計算 PR_raw 標準差
                    POS_std = np.std(pr_raw_values)# 計算 PR_normalized

                    if POS_std != 0:
                        POS_normalized = (POS_raw - POS_mean) / POS_std
                    else:
                        # 如果 PR_std 為零，請根據您的需求設置一個預設值，這裡設置為零
                        POS_normalized = 0

                    # df.loc[df.index[-1], 'S1'] = POS_matrix_operation_result[0]
                    # df.loc[df.index[-1], 'S2'] = POS_matrix_operation_result[1]
                    # df.loc[df.index[-1], 'Std_S1'] = std_s1
                    # df.loc[df.index[-1], 'Std_S2'] = std_s2
                    df.loc[df.index[-1], 'POS_raw'] = POS_raw
                    df.loc[df.index[-1], 'POS_DC'] = POS_mean
                    # df.loc[df.index[-1], 'POS_std'] = POS_std
                    # df.loc[df.index[-1], 'POS_normalized'] = POS_normalized
                    df.loc[df.index[-1], 'POS_AC'] = POS_raw - POS_mean
                    df.loc[df.index[-1], 'POS_normalized'] = POS_normalized
                    pr_normalized = df["POS_normalized"]
                    
                    
                    # 設計 80 階 FIR 濾波器的係數
                    order = 80
                    nyquist = 0.5 * 30  # Nyquist 頻率，這裡假設取樣頻率為 30 Hz
                    lowcut = 1
                    highcut = 1.67
                    cutoff = [lowcut / nyquist, highcut / nyquist]  # 截至頻率，轉換為正規化頻率

                    # 計算 FIR 濾波器係數
                    coefficients = firwin(order, cutoff, pass_zero=False)
                    # 應用 FIR 濾波器到 PR_normalized 資料
                    fir_filtered = lfilter(coefficients, 1.0, pr_normalized)
                    # 將處理完的訊號用 PR_filtered 表示
                    df["fir_filtered"] = fir_filtered
                    # POS Algorithm-----------------------------------------
                    
                # For CHROM Algorithm---------------------------------------
                chrom_std_x = np.std(CHROM_x_buffer, axis=0)
                chrom_std_y = np.std(CHROM_y_buffer, axis=0)
                chrom_alpha = chrom_std_x / chrom_std_y
                bvp_chrom = chrom_std_x + (chrom_alpha * chrom_std_y)
                # df.loc[df.index[-1], 'CHROM_X'] = CHROM_matrix_operation_result[0]
                # df.loc[df.index[-1], 'CHROM_Y'] = CHROM_matrix_operation_result[1]
                # df.loc[df.index[-1], 'CHROM_S'] = (CHROM_matrix_operation_result[0]/CHROM_matrix_operation_result[1]) - 1
                df.loc[df.index[-1], 'CHROM_raw'] = bvp_chrom
                # CHROM Algorithm---------------------------------------
                print(frame)
            if frame == 9000:
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame += 1


    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

def main():
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        video_path = r'Z:\SPO2_IRB_Lab_dataset\EE\Sub_21_1\Sub_21_1_face_30fps_2024-02-29 11_44_40.avi'
        csv_filename = r'Sub21-1.csv'

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f'Error: Unable to open the video file {video_path}')
            return

        process_video(cap, face_mesh, csv_filename)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
