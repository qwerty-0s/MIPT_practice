from image_embeddings import extract_embeddings
import numpy as np

# 1. Извлеките эмбеддинги
paths = ['/home/au/python/Mipt_project/EuroSAT/2750/Forest/Forest_1.jpg', 
         '/home/au/python/Mipt_project/EuroSAT/2750/Forest/Forest_2.jpg', 
         '/home/au/python/Mipt_project/EuroSAT/2750/River/River_1.jpg']
embeddings = extract_embeddings(paths)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim = cosine_similarity(embeddings[paths[0]], embeddings[paths[1]])
print(f"Сходство два изображения с лесом: {sim:.4f}")
sim = cosine_similarity(embeddings[paths[0]], embeddings[paths[2]])
print(f"Сходство изображения с лесом и изображением с рекой: {sim:.4f}")
