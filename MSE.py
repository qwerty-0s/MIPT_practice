import numpy as np
import cv2
import matplotlib.pyplot as plt

def mse(image1, image2):
    
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    diff = image1.astype(np.float32) - image2.astype(np.float32)
    diff_sq = diff ** 2
    diff_mn = np.mean(diff_sq) 
    return diff_mn


def ssim(image1, image2):
    # Стабилизация
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)

    # 1. Средние значения 
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    # 2. Квадраты средних и их произведение
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # 3. Дисперсии и ковариация 
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # 4. Формула SSIM
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return np.mean(ssim_map)

# Пример использования
img1 = cv2.imread("Mars1.jpg")
img2 = cv2.imread("MarsBlurred1.jpg")

print("MSE test:", mse(img1, img1))
print("MSE:", mse(img1, img2))
print("SSIM test:", ssim(img1, img1))
print("SSIM:", ssim(img1, img2))