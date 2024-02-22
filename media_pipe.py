import numpy as np
import cv2
import mediapipe as mp

def mask_nose_forehead(img, face_landmarks):
    bigger_forehead_indices = [109, 10, 338, 337, 151, 108]
    mask_forehead = np.zeros_like(img)
    pts_forehead = np.array([(int(face_landmarks.landmark[i].x * img.shape[1]), 
                    int(face_landmarks.landmark[i].y * img.shape[0])) for i in bigger_forehead_indices], np.int32)
    pts_forehead = pts_forehead.reshape((-1, 1, 2))
    cv2.fillPoly(mask_forehead, [pts_forehead], (255, 255, 255))
    forehead_masked = cv2.bitwise_and(img, mask_forehead)
    return forehead_masked

def process_video(cap, face_mesh):
    frame = 1
    second = 0
    
    while True:
        ret, img_rgb = cap.read()
        if not ret:
            break

        img_rgb2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb2)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                forehead_masked = mask_nose_forehead(img_rgb, face_landmarks)

        # cv2.imshow('Combined', combined_image)
        cv2.imshow('Forehead', forehead_masked)
        cv2.imshow('Original', img_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame += 1
        if frame % 30 == 0:
            second += 1
            print('second:', second)
            print('frame:', frame)


def main():
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        video_path = './video.avi'  # Update the path accordingly
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print('Error: Unable to open the video file.')
            return

        process_video(cap, face_mesh)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
