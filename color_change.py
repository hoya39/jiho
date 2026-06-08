import cv2
import os
import numpy as np

# 클래스 이름
class_name = "d8_4"

# 저장 위치
path = f"./dataset/train/{class_name}"

os.makedirs(path, exist_ok=True)

last = 1650

for count in range(last):

    readname = os.path.join(path, f"frame_{count}.jpg")
    img = cv2.imread(readname)

    # 이미지 없으면 건너뜀
    if img is None:
        print(f"Cannot read: {readname}")
        continue


    # 밝기 증가
    brighter = cv2.convertScaleAbs(img, alpha=1.1, beta=20)

    # 저장
    filename = os.path.join(path, f"frame_{count+last}.jpg")
    cv2.imwrite(filename, brighter)

    print("Saved:", filename)
