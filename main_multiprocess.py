import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from scipy.signal import butter, lfilter
import os
import multiprocessing

def mask_nose_forehead(img, face_landmarks):
    # 定義額頭區域的重要點索引
    bigger_forehead_indices = [109, 10, 338, 337, 151, 108]
    
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

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def process_video(cap, face_mesh, csv_filename):
    df = pd.DataFrame(columns=["R", "G", "B", "DC_red", "DC_green", "DC_blue", "AC_red", "AC_green", "AC_blue", "RR"])

    buffer_size = 100
    red_buffer = []
    green_buffer = []
    blue_buffer = []

    frame = 1
    while True:
        ret, img_rgb = cap.read()
        if not ret:
            break

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

                red_buffer.append(forehead_data["R"])
                green_buffer.append(forehead_data["G"])
                blue_buffer.append(forehead_data["B"])

                if len(red_buffer) > buffer_size:
                    red_buffer.pop(0)
                    green_buffer.pop(0)
                    blue_buffer.pop(0)

                if frame > buffer_size:
                    DC_red = np.mean(red_buffer)
                    DC_green = np.mean(green_buffer)
                    DC_blue = np.mean(blue_buffer)

                    # Apply bandpass filter to DC_red
                    fs = 30  # Adjust the sampling frequency accordingly
                    lowcut = 0.7
                    highcut = 3.0

                    filtered_DC_red = butter_bandpass_filter([DC_red], lowcut, highcut, fs)
                    filtered_DC_green = butter_bandpass_filter([DC_green], lowcut, highcut, fs)
                    filtered_DC_blue = butter_bandpass_filter([DC_blue], lowcut, highcut, fs)

                    AC_red = forehead_data["R"] - DC_red
                    AC_green = forehead_data["G"] - DC_green
                    AC_blue = forehead_data["B"] - DC_blue

                    RR = (AC_red/DC_red)/(AC_blue/DC_blue)

                    df.loc[df.index[-1], 'DC_red'] = DC_red
                    df.loc[df.index[-1], 'DC_green'] = DC_green
                    df.loc[df.index[-1], 'DC_blue'] = DC_blue
                    df.loc[df.index[-1], 'AC_red'] = AC_red
                    df.loc[df.index[-1], 'AC_green'] = AC_green
                    df.loc[df.index[-1], 'AC_blue'] = AC_blue
                    df.loc[df.index[-1], 'RR'] = RR
                    df.loc[df.index[-1], 'DC_red_filter'] = filtered_DC_red[0]
                    df.loc[df.index[-1], 'DC_green_filter'] = filtered_DC_green[0]
                    df.loc[df.index[-1], 'DC_blue_filter'] = filtered_DC_blue[0]

                print(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame += 1

    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")



def search_videos(base_folder):
    output_folder = 'video_output2'
    video_tasks = []

    for patient_folder in os.listdir(base_folder):
        patient_path = os.path.join(base_folder, patient_folder)
        if os.path.isdir(patient_path):
            video_path = os.path.join(patient_path, 'v1', 'source1', 'video.avi')
            if os.path.isfile(video_path):
                output_csv_name = f'{patient_folder}_v1_source1_output.csv'
                csv_filename = os.path.join(output_folder, output_csv_name)
                video_tasks.append((video_path, csv_filename))

    return video_tasks

def main():
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        base_folder = r'Z:\VIPL-HR_dataset\data'
        # base_folder = r'Z:\SPO2_IRB_Lab_dataset\Sub_9'
        # Z:\SPO2_IRB_Lab_dataset\Sub_9\Sub_9_face_30fps_2023-08-17 15_12_50.avi
        output_folder = 'video_output2'
        os.makedirs(output_folder, exist_ok=True)

        video_paths = search_videos(base_folder)

        for video_path, csv_filename in video_paths:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f'Error: Unable to open the video file {video_path}')
                continue

            process_video(cap, face_mesh, csv_filename)

            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
