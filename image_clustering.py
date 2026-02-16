import numpy as np
import cv2
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from numpy.lib.stride_tricks import sliding_window_view

import warnings
warnings.filterwarnings("ignore")


class ImageClusterer:
    
    def __init__(self, folder_path: str, ssim_threshold: float = 0.9):
        self.folder_path = folder_path
        self.ssim_threshold = ssim_threshold
        self.image_paths = []
        self.image_data = {}
        self.similarity_matrix = None
        self.clusters = {}
        
    def scan_images(self) -> List[str]:
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = []
        
        for file_path in Path(self.folder_path).rglob('*'):
            if file_path.suffix.lower() in supported_formats:
                image_paths.append(str(file_path))
        
        self.image_paths = sorted(image_paths)
        print(f"Найдено {len(self.image_paths)} изображений")
        return self.image_paths
    
    def preprocess_image(self, image_path: str) -> np.ndarray | None:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray.astype(np.float64)
        
        except Exception as e:
            print(f"Ошибка обработки {image_path}: {e}")
            return None
    
    def load_all_images(self) -> Dict[str, np.ndarray]:
        self.image_data = {}
        
        for image_path in self.image_paths:
            processed_img = self.preprocess_image(image_path)
            if processed_img is not None:
                self.image_data[image_path] = processed_img
        
        print(f"Загружено {len(self.image_data)} изображений")
        return self.image_data
    
    @staticmethod
    def ssim(image1: np.ndarray, image2: np.ndarray, window_size: int = 11) -> float:
        if image1.shape != image2.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = image1.astype(np.float64)
        img2 = image2.astype(np.float64)
        
        pad = window_size // 2
        img1_padded = np.pad(img1, pad, mode='reflect')
        img2_padded = np.pad(img2, pad, mode='reflect')
        
        windows1 = sliding_window_view(img1_padded, (window_size, window_size))
        windows2 = sliding_window_view(img2_padded, (window_size, window_size))
        
        mu1 = np.mean(windows1, axis=(2, 3))
        mu2 = np.mean(windows2, axis=(2, 3))
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = np.mean(windows1 ** 2, axis=(2, 3)) - mu1_sq
        sigma2_sq = np.mean(windows2 ** 2, axis=(2, 3)) - mu2_sq
        sigma12 = np.mean(windows1 * windows2, axis=(2, 3)) - mu1_mu2
        
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-10)
        
        return float(np.mean(ssim_map))
    
    def compute_similarity_matrix(self) -> np.ndarray:
        n = len(self.image_data)
        paths = list(self.image_data.keys())
        
        print(f"Вычисление матрицы сходства")
        self.similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    ssim_val = self.ssim(self.image_data[paths[i]], self.image_data[paths[j]])
                    self.similarity_matrix[i, j] = ssim_val
                    self.similarity_matrix[j, i] = ssim_val
                except Exception as e:
                    print(f"Ошибка: {e}")
                    self.similarity_matrix[i, j] = 0
                    self.similarity_matrix[j, i] = 0
            
            self.similarity_matrix[i, i] = 1.0
        
        print("Матрица сходства вычислена")
        return self.similarity_matrix
    
    def cluster_images(self) -> Dict[str, List[str]]:
        if self.similarity_matrix is None:
            raise ValueError("Сначала вычислите матрицу сходства")
        
        paths = list(self.image_data.keys())
        n = len(paths)
        used_indices = set()
        self.clusters = {}
        
        for i in range(n):
            if i in used_indices:
                continue
            
            similar_indices = np.where(
                (self.similarity_matrix[i, :] > self.ssim_threshold) & 
                (np.arange(n) != i)
            )[0]
            
            if len(similar_indices) > 0:
                original_path = paths[i]
                similar_paths = [paths[idx] for idx in similar_indices]
                
                self.clusters[original_path] = similar_paths
                
                used_indices.add(i)
                used_indices.update(similar_indices)
        
        print(f"Найдено {len(self.clusters)} групп")
        return self.clusters
    
    def export_json(self, output_path: str = "report.json") -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.clusters, f, indent=2, ensure_ascii=False)
        print(f"JSON: {output_path}")
    
    def export_csv(self, output_path: str = "report.csv") -> None:
        data = [
            {"Original": original, "Similars": ", ".join(similars)}
            for original, similars in self.clusters.items()
        ]
        pd.DataFrame(data).to_csv(output_path, index=False, encoding='utf-8')
        print(f"CSV: {output_path}")
    
    def export_xlsx(self, output_path: str = "report.xlsx") -> None:
        data = [
            {"Original": original, "Similars": ", ".join(similars)}
            for original, similars in self.clusters.items()
        ]
        pd.DataFrame(data).to_excel(output_path, index=False, engine='openpyxl')
        print(f"Excel: {output_path}")
    
    def export_all_reports(self, output_dir: str = ".") -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.export_json(os.path.join(output_dir, "report.json"))
        self.export_csv(os.path.join(output_dir, "report.csv"))
        self.export_xlsx(os.path.join(output_dir, "report.xlsx"))
    
    def run(self, output_dir: str = "clustering_results") -> Dict[str, List[str]]:
        self.scan_images()
        self.load_all_images()
        self.compute_similarity_matrix()
        self.cluster_images()
        self.export_all_reports(output_dir)
        
        print(f"\nРезультаты в: {output_dir}")
        return self.clusters


if __name__ == "__main__":
    folder_path = input("Путь к папке с изображениями: ").strip() or "."
    
    if not os.path.exists(folder_path):
        print(f"Папка не найдена")
        exit()

    clusterer = ImageClusterer(folder_path, ssim_threshold=0.9)
    clusters = clusterer.run()
    
    if clusters:
        print(f"\nВсего групп: {len(clusters)}")
        print(f"Средний размер: {np.mean([len(v) for v in clusters.values()]):.1f}")
