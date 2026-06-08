import cv2
import os

# # 클래스 이름
class_name = "d8_4"
video_path = f"./videos/{class_name}.mp4"

# # 저장 위치
save_path = f"./dataset/train/{class_name}"

os.makedirs(save_path, exist_ok=True)
cap = cv2.VideoCapture(video_path)
count = 1402

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # 3프레임마다 저장
    if frame_id % 3 == 0:
        filename = os.path.join(
            save_path,
            f"frame_{count}.jpg"
        )
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

cap.release()

print("Done")
