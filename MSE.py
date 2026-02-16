import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view

def mse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    diff = image1.astype(np.float32) - image2.astype(np.float32)
    diff_sq = diff ** 2
    diff_mn = np.mean(diff_sq) 
    return diff_mn


def ssim(image1, image2, window_size=11):

    # Константы для стабилизации
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    
    # Конвертируем в grayscale если цветное изображение
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Паддинг изображений
    pad = window_size // 2
    img1_padded = np.pad(img1, pad, mode='reflect')
    img2_padded = np.pad(img2, pad, mode='reflect')
    
    # Использование sliding_window_view для векторизации
    windows1 = sliding_window_view(img1_padded, (window_size, window_size))
    windows2 = sliding_window_view(img2_padded, (window_size, window_size))
    
    # Средние значения
    mu1 = np.mean(windows1, axis=(2, 3))
    mu2 = np.mean(windows2, axis=(2, 3))
    
    # Квадраты средних
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Дисперсии и ковариация (матрицы вычисляются на окнах)
    sigma1_sq = np.mean(windows1 ** 2, axis=(2, 3)) - mu1_sq
    sigma2_sq = np.mean(windows2 ** 2, axis=(2, 3)) - mu2_sq
    sigma12 = np.mean(windows1 * windows2, axis=(2, 3)) - mu1_mu2
    
    # Вычисление SSIM для каждого патча
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    # Избегаем деления на ноль
    ssim_map = np.divide(numerator, denominator, 
                         where=denominator!=0, 
                         out=np.ones_like(numerator))
    
    mean_ssim = np.mean(ssim_map)
    
    return mean_ssim


if __name__ == "__main__":
    img1 = cv2.imread("Mars1.jpg")
    img2 = cv2.imread("MarsBlurred1.jpg")

    if img1 is not None and img2 is not None:
        print("MSE test:", mse(img1, img1))
        print("MSE:", mse(img1, img2))
        
        ssim_test = ssim(img1, img1)
        print("SSIM test (mean):", ssim_test)
        
        ssim_mean = ssim(img1, img2)
        print("SSIM (mean):", ssim_mean)
    else:
        print("Изображения не найдены")