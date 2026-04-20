from text_embeddings import extract_embeddings, save_embeddings, load_embeddings
import numpy as np
from typing import List, Tuple


def build_database(texts: List[str], save_path: str = "text_embeddings.npz") -> dict:

    embeddings = extract_embeddings(texts)
    save_embeddings(embeddings, save_path)
    return embeddings


def search(
    query: str,
    embeddings: dict,
    top_k: int = 3,
) -> List[Tuple[str, float]]:

    # Кодируем запрос тем же энкодером
    query_vec = extract_embeddings([query])[query]          # (384,)

    # Считаем косинусное сходство со всей базой векторизовано
    texts = list(embeddings.keys())
    matrix = np.stack([embeddings[t] for t in texts])      # (N, 384)

    scores = matrix @ query_vec                             # (N,)

    # Топ-K индексов
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(texts[i], float(scores[i])) for i in top_indices]


if __name__ == "__main__":

    database_texts = [
        "A dense green forest with tall trees and thick vegetation.",
        "A woodland area filled with coniferous trees and a lush canopy.",
        "A winding river flowing through an open valley with sandy banks.",
        "A wide river delta seen from above with muddy water and islands.",
        "An open field of crops stretching to the horizon under a blue sky.",
        "A residential area with rows of houses and streets seen from above.",
        "A highway interchange with multiple lanes and overpasses.",
        "A snow-covered mountain peak with rocky cliffs and glaciers.",
    ]

    embeddings = build_database(database_texts)

    queries = [
        "trees and green forest",
        "water stream",
        "urban infrastructure",
    ]

    for query in queries:
        print(f"\nЗапрос: '{query}'")
        results = search(query, embeddings, top_k=3)
        for rank, (text, score) in enumerate(results, 1):
            print(f"  {rank}. [{score:.4f}] {text}")