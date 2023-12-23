import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import lfilter, find_peaks, butter

# model for face detection
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

def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)  # make sure output folder exists
    output_csv_path = os.path.join(output_folder, f"{video_name}_output.csv")

    df = pd.DataFrame(columns=["R", "G", "B", "DC_red", "DC_green", "DC_blue", "AC_red", "AC_green", "AC_blue", "RR", "SPO2"])

    frame_count = 0
    buffer_size = 10
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
            x0 = max(0, x - 10)
            y0 = max(0, y - 10)
            x1 = min(frame.shape[1], x + w + 10)
            y1 = min(frame.shape[0], y + h + 10)
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

            if len(red_buffer or green_buffer or blue_buffer ) > buffer_size:
                red_buffer.pop(0)
                green_buffer.pop(0)
                blue_buffer.pop(0)


                AC_red = r-DC_red
                AC_green = g-DC_green
                AC_blue = b-DC_blue

                df.loc[df.index[-1], 'AC_red'] = AC_red
                df.loc[df.index[-1], 'AC_green'] = AC_green
                df.loc[df.index[-1], 'AC_blue'] = AC_blue

                df['AC_red_filtered'] = butter_bandpass_filter(df['AC_red'].values, lowcut, highcut, fs)
                df['AC_green_filtered'] = butter_bandpass_filter(df['AC_green'].values, lowcut, highcut, fs)
                df['AC_blue_filtered'] = butter_bandpass_filter(df['AC_blue'].values, lowcut, highcut, fs)

                AC_red_filtered = df['AC_red_filtered'].values
                AC_green_filtered = df['AC_green_filtered'].values
                AC_blue_filtered = df['AC_blue_filtered'].values

                # RR = (AC_red/DC_red) / (AC_blue/DC_blue)
                RR = (np.mean(AC_red_filtered)/DC_red) / (np.mean(AC_blue_filtered)/DC_blue)
                df.loc[df.index[-1], 'RR'] = RR

                SPO2 = 95 - 5 * RR
                df.loc[df.index[-1], 'SPO2'] = SPO2

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            frame_count += 1
            print(frame_count)
        # to limit the frame count
        if frame_count == 5400:
                break

        lower_skin = np.array([0, 20, 70], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        face_mask = np.zeros_like(hsv_frame[:, :, 0])
        for (x, y, w, h) in detected_faces:
            x0 = max(0, x - 10)
            y0 = max(0, y - 10)
            x1 = min(frame.shape[1], x + w + 10)
            y1 = min(frame.shape[0], y + h + 10)
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

    df.to_csv(output_csv_path, index=False)
    cap.release()
    cv2.destroyAllWindows()

def process_videos_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".avi"):
            video_path = os.path.join(folder_path, filename)
            process_video(video_path)

def main():
    folder_path = r'.\data'  # saving all the videos in the data folder
    process_videos_in_folder(folder_path)

if __name__ == '__main__':
    main()
