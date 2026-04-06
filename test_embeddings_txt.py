from text_embeddings import extract_embeddings
import numpy as np

# 1. Извлекаем эмбеддинги
texts = [
    "A dense green forest with tall trees and thick vegetation covering the landscape.",
    "A woodland area filled with coniferous trees and a lush green canopy from above.",
    "A winding river flowing through an open valley with sandy banks and clear water.",
]
embeddings = extract_embeddings(texts)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


sim = cosine_similarity(embeddings[texts[0]], embeddings[texts[1]])
print(f"Сходство два описания леса:           {sim:.4f}")

sim = cosine_similarity(embeddings[texts[0]], embeddings[texts[2]])
print(f"Сходство описания леса и реки:        {sim:.4f}")