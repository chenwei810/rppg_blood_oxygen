import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from scipy.signal import savgol_filter

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
    df = pd.DataFrame(columns=["R", "R_filter", "G", "G_filter", "B", "B_filter", "AC_red", "DC_red", "AC_green", "DC_green", "AC_blue", "DC_blue", "RR", "POS_AC", "CHROM_AC"])

    buffer_size = 30
    red_buffer, green_buffer, blue_buffer = [], [], []
    AC_red_buffer, AC_green_buffer, AC_blue_buffer = [], [], []

    POS_s1_buffer, POS_s2_buffer = [], []
    POS_raw_buffer = []

    CHROM_x_buffer, CHROM_y_buffer = [], []
    CHROM_raw_buffer = []

    frame_count = 0

    frame = 1
    while True:
        ret, img_rgb = cap.read()
        if not ret:
            break
        frame_count += 1

        img_rgb2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # 對 RGB 訊號應用 Savitzky-Golay 濾波器
        img_rgb2[:, :, 0] = savgol_filter(img_rgb2[:, :, 0], 30, 4)  # Red channel
        img_rgb2[:, :, 1] = savgol_filter(img_rgb2[:, :, 1], 30, 4)  # Green channel
        img_rgb2[:, :, 2] = savgol_filter(img_rgb2[:, :, 2], 30, 4)  # Blue channel

        results = face_mesh.process(img_rgb2)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                forehead_data = mask_nose_forehead(img_rgb, face_landmarks)

                df = pd.concat([df, pd.DataFrame({
                    "R": [forehead_data["R"]],
                    "G": [forehead_data["G"]],
                    "B": [forehead_data["B"]],
                    "R_filter": [img_rgb2[:, :, 0].mean()],
                    "G_filter": [img_rgb2[:, :, 1].mean()],
                    "B_filter": [img_rgb2[:, :, 2].mean()]
                })], ignore_index=True)

                # POS Algorithm-----------------------------------------
                POS_matrix_operation_result = np.array([
                    [0, 1, -1],
                    [-2, 1, 1]
                ]).dot([img_rgb2[:, :, 0].mean(), img_rgb2[:, :, 1].mean(), img_rgb2[:, :, 2].mean()])
                
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
                ]).dot([img_rgb2[:, :, 0].mean(), img_rgb2[:, :, 1].mean(), img_rgb2[:, :, 2].mean()])
                CHROM_x_buffer.append(CHROM_matrix_operation_result[0])
                CHROM_y_buffer.append(CHROM_matrix_operation_result[1])

                if len(CHROM_x_buffer) > buffer_size:
                    CHROM_x_buffer.pop(0)
                    CHROM_y_buffer.pop(0)

                # CHROM Algorithm END---------------------------------------

                red_buffer.append(img_rgb2[:, :, 0].mean())
                green_buffer.append(img_rgb2[:, :, 1].mean())
                blue_buffer.append(img_rgb2[:, :, 2].mean())

                if len(red_buffer) > buffer_size:
                    red_buffer.pop(0)
                    green_buffer.pop(0)
                    blue_buffer.pop(0)

                if frame > buffer_size:

                    # For SPO2 calculation
                    DC_red = np.mean(red_buffer)
                    DC_green = np.mean(green_buffer)
                    DC_blue = np.mean(blue_buffer)

                    AC_red = img_rgb2[:, :, 0].mean() - DC_red
                    AC_green = img_rgb2[:, :, 1].mean() - DC_green
                    AC_blue = img_rgb2[:, :, 2].mean() - DC_blue

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

                    RR = np.log((max_AC_red-min_AC_red)) / np.log((max_AC_blue-min_AC_blue))

                    df.loc[df.index[-1], 'AC_red'] = max_AC_red - min_AC_red
                    df.loc[df.index[-1], 'DC_red'] = DC_red
                    df.loc[df.index[-1], 'AC_green'] = max_AC_green - min_AC_green
                    df.loc[df.index[-1], 'DC_green'] = DC_green
                    df.loc[df.index[-1], 'AC_blue'] = max_AC_blue - min_AC_blue
                    df.loc[df.index[-1], 'DC_blue'] = DC_blue
                    df.loc[df.index[-1], 'RR'] = RR

                    # For POS Algorithm-----------------------------------------
                    std_s1 = np.std(POS_s1_buffer)
                    std_s2 = np.std(POS_s2_buffer)
                    alpha = std_s1 / std_s2
                    POS_raw = std_s1 + (alpha * std_s2)
                    if len(POS_raw_buffer) > buffer_size:
                        POS_raw_buffer.pop(0)
                    POS_raw_buffer.append(POS_raw)
                    max_POS = np.max(POS_raw_buffer)
                    min_POS = np.min(POS_raw_buffer)

                    # if (max_POS - min_POS) == 0:
                    #     POS_AC = 0
                    # else:
                    POS_AC = max_POS - min_POS
                    # df.loc[df.index[-1], 'POS'] = POS_raw
                    df.loc[df.index[-1], 'POS_AC'] = POS_AC
                    # POS Algorithm END-----------------------------------------

                # For CHROM Algorithm---------------------------------------
                chrom_std_x = np.std(CHROM_x_buffer, axis=0)
                chrom_std_y = np.std(CHROM_y_buffer, axis=0)
                chrom_alpha = chrom_std_x / chrom_std_y
                bvp_chrom = chrom_std_x + (chrom_alpha * chrom_std_y)

                if len(CHROM_raw_buffer) > buffer_size:
                    CHROM_raw_buffer.pop(0)
                CHROM_raw_buffer.append(bvp_chrom)
                max_CHROM = np.max(CHROM_raw_buffer)
                min_CHROM = np.min(CHROM_raw_buffer)

                # if (max_CHROM - min_CHROM) == 0:
                #     CHROM_AC = 0
                # else:
                CHROM_AC = max_CHROM - min_CHROM

                # df.loc[df.index[-1], 'CHROM'] = bvp_chrom
                df.loc[df.index[-1], 'CHROM_AC'] = CHROM_AC
                # CHROM Algorithm END---------------------------------------

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
        video_path = [
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_9_0\Sub_9_0_face_30fps_2024-03-04 19_35_10.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_18_0\Sub_18_0_face_30fps_2024-03-01 13_21_30.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_18_1\Sub_18_1_face_30fps_2024-03-01 13_27_50.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_19_0\Sub_19_0_face_30fps_2024-03-01 14_43_30.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_19_1\Sub_19_1_face_30fps_2024-03-01 14_49_20.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_21_1\Sub_21_1_face_30fps_2024-02-29 11_44_40.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_21_2\Sub_21_2_face_30fps_2024-02-29 11_51_30.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_22_1\Sub_22_1_face_30fps_2024-02-29 14_45_20.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_22_2\Sub_22_2_face_30fps_2024-02-29 14_51_50.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_23_1\Sub_23_1_face_30fps_2024-02-29 15_39_10.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_23_2\Sub_23_2_face_30fps_2024-02-29 15_50_40.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_24_0\Sub_24_0_face_30fps_2024-03-01 11_27_20.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_24_1\Sub_24_1_face_30fps_2024-03-01 11_35_30.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_25_0\Sub_25_0_face_30fps_2024-03-01 15_23_40.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_25_1\Sub_25_1_face_30fps_2024-03-01 15_29_50.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_26_0\Sub_26_0_face_30fps_2024-03-04 14_49_40.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_26_1\Sub_26_1_face_30fps_2024-03-04 14_55_40.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_29_0\Sub_29_0_face_30fps_2024-03-05 17_59_00.avi',
            'Z:\SPO2_IRB_Lab_dataset\EE\Sub_30_0\Sub_30_0_face_30fps_2024-03-05 19_23_50.avi',
        ]

        csv_filename = [
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

        for video_path, csv_filename in zip(video_path, csv_filename):
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f'Error: Unable to open the video file {video_path}')
                return
            process_video(cap, face_mesh, csv_filename)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
