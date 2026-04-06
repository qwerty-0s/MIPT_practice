import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")


# ── Загружаем модель один раз при импорте модуля ──────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Загрузка модели sentence-transformer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Режим inference
print("Модель загружена")


#Mean Pooling (стандартный способ получить sentence embedding)
def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state          # (B, T, H)
    mask_expanded = attention_mask.unsqueeze(-1).float()       # (B, T, 1)
    summed = (token_embeddings * mask_expanded).sum(dim=1)     # (B, H)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)          # (B, 1)
    return summed / counts                                     # (B, H)


def extract_embeddings(
    texts: List[str],
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Извлекает эмбеддинги из списка текстовых описаний.

    Args:
        texts:      Список строк (описаний изображений).
        batch_size: Размер батча при инференсе.

    Returns:
        Словарь {текст: эмбеддинг (numpy array, 384 элемента, L2-нормирован)}.
    """
    embeddings: Dict[str, np.ndarray] = {}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = model(**encoded)

        vecs = _mean_pooling(output, encoded["attention_mask"])  # (B, 384)

        # L2-нормировка — аналог нормировки перед cosine similarity
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        vecs_np = vecs.numpy()

        for text, vec in zip(batch, vecs_np):
            embeddings[text] = vec

        processed = min(i + batch_size, len(texts))
        if processed % 100 == 0 or processed == len(texts):
            print(f"Обработано {processed}/{len(texts)} описаний")

    print(f"\nВсего извлечено эмбеддингов: {len(embeddings)}")
    return embeddings


def scan_folder(folder_path: str) -> List[str]:
    """
    Сканирует папку и возвращает список текстовых описаний из .txt файлов.
    Каждый файл — одно описание (одна строка или несколько, объединяются).
    """
    texts: List[str] = []

    for file_path in Path(folder_path).rglob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                texts.append(text)
        except Exception as e:
            print(f"Ошибка чтения {file_path}: {e}")

    print(f"Найдено {len(texts)} текстовых описаний")
    return texts


def save_embeddings(
    embeddings: Dict[str, np.ndarray],
    output_path: str = "text_embeddings.npz",
) -> None:
    """Сохраняет эмбеддинги в файл .npz"""
    # Ключи npz не могут содержать спецсимволы → кодируем индексом
    keys = list(embeddings.keys())
    vecs = np.stack([embeddings[k] for k in keys])          # (N, 384)

    np.savez_compressed(
        output_path,
        texts=np.array(keys, dtype=object),
        vectors=vecs,
    )
    print(f"Эмбеддинги сохранены в {output_path}")


def load_embeddings(
    input_path: str = "text_embeddings.npz",
) -> Dict[str, np.ndarray]:
    """Загружает эмбеддинги из файла .npz"""
    data = np.load(input_path, allow_pickle=True)
    texts  = data["texts"].tolist()
    vectors = data["vectors"]
    embeddings = {text: vec for text, vec in zip(texts, vectors)}
    print(f"Загружено {len(embeddings)} эмбеддингов")
    return embeddings


if __name__ == "__main__":
    folder_path = input("Путь к папке с .txt описаниями: ").strip() or "."

    texts = scan_folder(folder_path)

    if texts:
        embeddings = extract_embeddings(texts)
        save_embeddings(embeddings, "text_embeddings.npz")

        sample_vec = list(embeddings.values())[0]
        print(f"\nРазмерность эмбеддинга: {sample_vec.shape}")
    else:
        print("Текстовые описания не найдены")