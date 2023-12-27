import os
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from scipy.signal import lfilter, find_peaks, butter

# Model for face detection
prototxt_rgb = r".\model\rgb.prototxt"
caffemodel_rgb = r".\model\rgb.caffemodel"
net_rgb = cv2.dnn.readNetFromCaffe(
    prototxt=prototxt_rgb, caffeModel=caffemodel_rgb)

def detect_faces(frame, min_confidence=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net_rgb.setInput(blob)
    detections = net_rgb.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x0, y0, x1, y1) = box.astype("int")
        faces.append((x0, y0, x1 - x0, y1 - y0))
    return faces

def extract_face_color(frame, x, y, w, h):
    face_roi = frame[y:y+h, x:x+w]
    mean_color = np.mean(face_roi, axis=(0, 1))
    return mean_color

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def find_peak_distance(data, threshold=0.5, lowcut=0.7, highcut=5, fs=30):
    filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs)
    peaks, _ = find_peaks(filtered_data, height=threshold)
    distances = np.diff(peaks)
    return distances

def process_video(video_path, df, output_folder, video_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    buffer_size = 100
    red_buffer = []
    green_buffer = []
    blue_buffer = []

    lowcut = 0.7
    highcut = 5
    fs = 30

    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        detected_faces = detect_faces(frame)

        for (x, y, w, h) in detected_faces:
            x0, y0, x1, y1 = max(0, x - 10), max(0, y - 10), min(frame.shape[1], x + w + 10), min(frame.shape[0], y + h + 10)
            face_color = extract_face_color(frame, x0, y0, x1 - x0, y1 - y0)
            r, g, b = face_color
            red_buffer.append(r)
            green_buffer.append(g)
            blue_buffer.append(b)

            df = pd.concat([df, pd.DataFrame({"R": [r], "G": [g], "B": [b], "DC_red": [0], "DC_green": [0], "DC_blue": [0], "AC_red": [0], "AC_green": [0], "AC_blue": [0], "RR": [0], "SPO2": [0]})], ignore_index=True)

            DC_red = np.mean(red_buffer)
            DC_green = np.mean(green_buffer)
            DC_blue = np.mean(blue_buffer)

            df.loc[df.index[-1], 'DC_red'] = DC_red
            df.loc[df.index[-1], 'DC_green'] = DC_green
            df.loc[df.index[-1], 'DC_blue'] = DC_blue

            if len(red_buffer) > buffer_size:
                red_buffer.pop(0)
                green_buffer.pop(0)
                blue_buffer.pop(0)

                AC_red, AC_green, AC_blue = r - DC_red, g - DC_green, b - DC_blue

                df.loc[df.index[-1], 'AC_red'] = AC_red
                df.loc[df.index[-1], 'AC_green'] = AC_green
                df.loc[df.index[-1], 'AC_blue'] = AC_blue

                df['AC_red_filtered'] = butter_bandpass_filter(df['AC_red'].values, lowcut, highcut, fs)
                df['AC_green_filtered'] = butter_bandpass_filter(df['AC_green'].values, lowcut, highcut, fs)
                df['AC_blue_filtered'] = butter_bandpass_filter(df['AC_blue'].values, lowcut, highcut, fs)

                AC_red_filtered, AC_green_filtered, AC_blue_filtered = df['AC_red_filtered'].values, df['AC_green_filtered'].values, df['AC_blue_filtered'].values

                RR = (np.mean(AC_red_filtered) / DC_red) / (np.mean(AC_blue_filtered) / DC_blue)
                df.loc[df.index[-1], 'RR'] = RR

                SPO2 = 96.2200388975547 + (-0.012162390435191146) * RR
                df.loc[df.index[-1], 'SPO2'] = SPO2

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            frame_count += 1
            print(frame_count)

        if frame_count == 5400:
            break

        lower_skin = np.array([0, 20, 70], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        face_mask = np.zeros_like(hsv_frame[:, :, 0])
        for (x, y, w, h) in detected_faces:
            x0, y0, x1, y1 = max(0, x - 10), max(0, y - 10), min(frame.shape[1], x + w + 10), min(frame.shape[0], y + h + 10)
            face_mask[y0:y1, x0:x1] = 255

        kernel_erode = np.ones((17, 17), np.uint8)
        face_mask = cv2.erode(face_mask, kernel_erode, iterations=2)

        kernel_dilate = np.ones((1, 1), np.uint8)
        face_mask = cv2.dilate(face_mask, kernel_dilate, iterations=2)

        skin_mask = cv2.bitwise_and(cv2.inRange(hsv_frame, lower_skin, upper_skin), face_mask)
        skin_extracted = cv2.bitwise_and(frame, frame, mask=skin_mask)

        cv2.imshow("Real-Time Face Detection", frame)
        cv2.imshow("Skin Extracted", skin_extracted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()

    # Save the DataFrame to CSV after processing each video
    # output_csv_path = os.path.join(output_folder, f"{video_name}_output.csv")
    output_csv_path = os.path.join(output_folder, f"{video_name}_output.csv")
    df.to_csv(output_csv_path, index=False)
    cv2.destroyAllWindows()

def process_video_wrapper(args):
    video_path, df, output_folder, video_name = args
    process_video(video_path, df, output_folder, video_name)

def main():
    # base_folder = 'VIPL-HR_dataset/data/'  # specify the path to your base folder
    base_folder = r'Z:\VIPL-HR_dataset\data'
    output_folder = 'video_output'
    os.makedirs(output_folder, exist_ok=True)

    # Use multiprocessing to parallelize video processing
    pool = multiprocessing.Pool(processes=5)  # Adjust the number of processes as needed

    video_tasks = []
    for p_folder in os.listdir(base_folder):
        p_path = os.path.join(base_folder, p_folder)
        if os.path.isdir(p_path):
            v_path = os.path.join(p_path, 'v1')  # 直接指定為 'v1'
            if os.path.isdir(v_path):
                for source_folder in os.listdir(v_path):
                    source_path = os.path.join(v_path, source_folder)
                    if os.path.isdir(source_path):
                        video_path = os.path.join(source_path, "video.avi")
                        combined_df = pd.DataFrame(columns=["R", "G", "B", "DC_red", "DC_green", "DC_blue", "AC_red", "AC_green", "AC_blue", "RR", "SPO2"])
                        video_tasks.append((video_path, combined_df, output_folder, f"{p_folder}_{source_folder}"))

    pool.map(process_video_wrapper, video_tasks)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()

    main()