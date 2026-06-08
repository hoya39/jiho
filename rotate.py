import cv2
import os
import numpy as np
import random

# =========================
# 클래스 이름
# =========================

class_name = "d8_4"

# =========================
# 이미지 폴더
# =========================

path = f"./dataset/train/{class_name}"

os.makedirs(path, exist_ok=True)

# =========================
# 이미지 목록
# =========================

image_list = os.listdir(path)

count = 0
i = 0

for image_name in image_list:

    # jpg 파일만 처리
    if not image_name.endswith(".jpg"):
        continue

    # 이미 회전된 이미지 제외
    if image_name.startswith("rot_"):
        continue

    readname = os.path.join(
        path,
        image_name
    )

    img = cv2.imread(readname)

    # 이미지 읽기 실패
    if img is None:
        print(f"Cannot read: {readname}")
        continue

    # =========================
    # 회전 각도 랜덤 생성
    # =========================

    angle = random.uniform(-15, 15)

    # =========================
    # 회전 행렬 생성
    # =========================

    h, w, _ = img.shape

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    # =========================
    # 이미지 회전
    # =========================

    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    # =========================
    # 저장
    # =========================

    save_name = os.path.join(
        path,
        f"rot_{i}.jpg"
    )

    cv2.imwrite(save_name, rotated)

    print(f"Saved: {save_name}")
    i += 1
    count += 3

print("Done")