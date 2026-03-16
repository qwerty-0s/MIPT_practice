import numpy as np
import cv2
import pandas as pd
import os
# Импортируем твои модули (убедись, что они лежат в той же папке)
from TFIDF import text_process, calculate_tfidf
from MSE import ssim

def cosine_similarity(v1, v2):
    """Вычисляет косинусное сходство между двумя векторами."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

def find_unique_images(query, df, image_folder, ssim_threshold=0.85, top_k=5):
    """
    1. Ищет изображения по текстовому описанию (TF-IDF).
    2. Фильтрует визуально похожие (SSIM).
    """
    # --- ЭТАП 1: Текстовый поиск (TF-IDF) ---
    descriptions = df['caption'].tolist()
    all_texts = descriptions + [query]
    
    # Расчет TF-IDF (используем твою функцию)
    tfidf_matrix = np.array(calculate_tfidf(all_texts))
    
    query_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]
    
    scores = []
    for i, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(query_vector, doc_vec)
        if score > 0:
            scores.append((i, score))
    
    # Сортируем: сначала самые релевантные тексты
    scores.sort(key=lambda x: x[1], reverse=True)
    
   
    results = []
    seen_images_data = [] 
    seen_paths = set()    

    for idx, text_score in scores:
        img_name = df.iloc[idx]['image']
        img_path = os.path.join(image_folder, img_name)
        
        # Если этот файл уже прошел проверку и добавлен в результаты, пропускаем
        if img_path in seen_paths:
            continue

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Превращаем в ЧБ для SSIM
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        is_duplicate = False
        for seen_gray in seen_images_data:
            if ssim(gray, seen_gray) > ssim_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            results.append({
                'path': img_path,
                'score': round(text_score, 4),
                'caption': df.iloc[idx]['caption']
            })
            seen_images_data.append(gray)
            seen_paths.add(img_path)
            
        if len(results) >= top_k:
            break

    return results

if __name__ == "__main__":
    # --- НАСТРОЙКИ ---
    PATH_TO_CSV = "/home/au/python/Mipt_project/test finding/captions.txt" 
    PATH_TO_IMAGES = "/home/au/python/Mipt_project/test finding/Images"

    if not os.path.exists(PATH_TO_CSV):
        print(f"Файл {PATH_TO_CSV} не найден!")
        exit()

    try:
        df_real = pd.read_csv(PATH_TO_CSV)
        df_real = df_real[0:1000]

        print(f"Загружено {len(df_real)} строк. Уникальных фото в списке: {df_real['image'].nunique()}")
    except Exception as e:
        print(f"Ошибка при чтении CSV: {e}")
        exit()

    query = input("Введите поисковый запрос: ")
    threshold_input = input("Порог сходства SSIM (0.1 - 1.0, по умолчанию 0.85): ")
    user_ssim = float(threshold_input) if threshold_input.strip() else 0.85
    
    print("\nИщем...")
    findings = find_unique_images(query, df_real, PATH_TO_IMAGES, ssim_threshold=user_ssim)

    print("\n" + "="*50)
    print(f"РЕЗУЛЬТАТЫ ПО ЗАПРОСУ: '{query}'")
    print("="*50)
    
    if not findings:
        print("Ничего не найдено. Попробуй изменить запрос или снизить порог SSIM.")
    else:
        for i, res in enumerate(findings, 1):
            print(f"{i}. [Score: {res['score']}]")
            print(f"   Файл: {os.path.basename(res['path'])}")
            print(f"   Описание: {res['caption']}\n")