import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")


# Загружаем модель один раз при импорте модуля
print("Загрузка модели ResNet18...")
model = models.resnet18(pretrained=True)
# Удаляем последний классификационный слой
model = nn.Sequential(*list(model.children())[:-1])
model.eval()  # Режим inference
print("Модель загружена")

# Трансформации для предобработки изображений
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_embeddings(image_paths: List[str]) -> Dict[str, np.ndarray]:
    """
    Извлекает эмбеддинги из списка изображений
    
    Args:
        image_paths: Список путей к изображениям
        
    Returns:
        Словарь {путь_к_изображению: эмбеддинг (numpy array 512 элементов)}
    """
    embeddings = {}
    
    for i, image_path in enumerate(image_paths):
        try:
            # Загружаем и обрабатываем изображение
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # Добавляем batch dimension
            
            # Извлекаем эмбеддинг
            with torch.no_grad():
                embedding = model(img_tensor)
            
            # Конвертируем в numpy array и flatten
            embedding_np = embedding.numpy().flatten()
            embeddings[image_path] = embedding_np
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(image_paths)} изображений")
            
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            continue
    
    print(f"\nВсего извлечено эмбеддингов: {len(embeddings)}")
    return embeddings


def scan_folder(folder_path: str) -> List[str]:
    """Сканирует папку и возвращает список путей к изображениям"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_paths = []
    
    for file_path in Path(folder_path).rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            image_paths.append(str(file_path))
    
    print(f"Найдено {len(image_paths)} изображений")
    return sorted(image_paths)


def save_embeddings(embeddings: Dict[str, np.ndarray], output_path: str = "embeddings.npz"):
    """Сохраняет эмбеддинги в файл"""
    np.savez_compressed(output_path, **embeddings)
    print(f"Эмбеддинги сохранены в {output_path}")


def load_embeddings(input_path: str = "embeddings.npz") -> Dict[str, np.ndarray]:
    """Загружает эмбеддинги из файла"""
    data = np.load(input_path, allow_pickle=True)
    embeddings = {key: data[key] for key in data.files}
    print(f"Загружено {len(embeddings)} эмбеддингов")
    return embeddings


if __name__ == "__main__":
    # Пример использования
    folder_path = input("Путь к папке с изображениями: ").strip() or "."
    
    # Сканируем папку
    image_paths = scan_folder(folder_path)
    
    if image_paths:
        # Извлекаем эмбеддинги
        embeddings = extract_embeddings(image_paths)
        
        # Сохраняем результат
        save_embeddings(embeddings, "embeddings.npz")
        
        print(f"\nРазмерность эмбеддинга: {list(embeddings.values())[0].shape}")
    else:
        print("Изображения не найдены")