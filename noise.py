import cv2
import os
import numpy as np

class_name = "d8_4"

path = f"./dataset/train/{class_name}"

os.makedirs(path, exist_ok=True)

image_list = os.listdir(path)

count = 0
i = 0

for image_name in image_list:

    if not image_name.endswith(".jpg"):
        continue

    # 원본 noise 파일 제외
    if image_name.startswith("noise_"):
        continue

    readname = os.path.join(
        path,
        image_name
    )

    img = cv2.imread(readname)

    if img is None:
        print(f"Cannot read: {readname}")
        continue

    # Gaussian noise 생성
    noise = np.random.normal(
        0,
        15,
        img.shape
    ).astype(np.int16)

    noisy_img = img.astype(np.int16) + noise

    noisy_img = np.clip(
        noisy_img,
        0,
        255
    ).astype(np.uint8)

    # noise 이미지 저장
    save_name = os.path.join(
        path,
        f"noise_{i}.jpg"
    )

    cv2.imwrite(save_name, noisy_img)

    print(f"Saved: {save_name}")

    i += 1
    count += 3

print("Done")