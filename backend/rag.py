import re
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

print(f"read vacancies")

df_vacancies = pd.read_parquet('./data_artefacts/vacancy_final.parquet')

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_text(text: str) -> list:
    """Токенизация текста для BM25 без использования NLTK"""
    normalized = normalize_text(text)
    
    tokens = re.split(r'[\s\W]+', normalized)
    
    # Фильтруем пустые строки и слишком короткие токены
    tokens = [token for token in tokens if len(token) > 2]
    return tokens

# Подготовка текстов вакансий
vacancy_texts = []
tokenized_corpus = []

for _, row in df_vacancies.iterrows():
    # Объединяем все текстовые поля вакансии
    full_text = f"{row['title']} {row['company']} {', '.join(row['skills'])} {row['experience']} {row['keywords']}"
    
    vacancy_texts.append(normalize_text(full_text))
    tokenized_corpus.append(tokenize_text(full_text))

print(f"prepare BM25 index")

# Создание BM25 индекса
bm25 = BM25Okapi(tokenized_corpus)

print(f"BM25 index created")

print(f"graph init")

G = nx.DiGraph()
position_nodes = []

for i, row in df_vacancies.iterrows():
    pos_node = row["title"]
    position_nodes.append(pos_node)
    
    # Вершины вакансии
    G.add_node(pos_node, type="position", vacancy_id=row["vacancy_id"], 
               company=row["company"], experience=row["experience"],
               salary=row["salary_str"], industry=row["industry"],
               requirements=row["keywords"],
               bm25_index=i)  # Сохраняем индекс для BM25
    
    if row["company"]:
        G.add_node(row["company"], type="company")
        G.add_edge(pos_node, row["company"])
    if row["experience"]:
        G.add_node(row["experience"], type="level")
        G.add_edge(pos_node, row["experience"])
    if row["industry"]:
        G.add_node(row["industry"], type="domain")
        G.add_edge(pos_node, row["industry"])

    # Навыки
    skills_arr = row["skills"]
    if isinstance(skills_arr, (list, np.ndarray)):
        for skill in skills_arr:
            skill = str(skill).strip()
            if skill:
                G.add_node(skill, type="skill")
                G.add_edge(pos_node, skill)  # position → skill
                G.add_edge(skill, pos_node)  # skill → position

print(f"graph done")

def recommend_vacancies(user_text, top_k=5, top_career=1, min_skill_freq=2, top_skills=10):
    """
    Рекомендация вакансий на основе BM25
    """
    # Токенизируем пользовательский запрос
    user_tokens = tokenize_text(user_text)
    
    # Получаем BM25 скоры для всех документов
    bm25_scores = bm25.get_scores(user_tokens)
    
    # Получаем топ-K индексов с наивысшими скорами
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    
    recommendations = []
    career_paths = set()
    all_neighbor_skills = []
    
    for idx in top_indices:
        node = position_nodes[idx]
        n_data = G.nodes[node]
        
        skills = [s for s in G.successors(node) if G.nodes[s]["type"]=="skill"]
        
        # BM25 скор как мера релевантности
        bm25_score = float(bm25_scores[idx])
        
        recommendations.append({
            "title": node,
            "company": n_data["company"],
            "experience": n_data["experience"],
            "salary": n_data["salary"],
            "industry": n_data["industry"],
            "skills": skills,
            "requirements":n_data["requirements"],
            "bm25_score": bm25_score,
            "similarity_score": min(bm25_score / max(bm25_scores), 1.0)  # Нормализованный скор
        })
        
        # Поиск похожих позиций через навыки
        for skill in skills:
            neighbors = [pos for pos in G.successors(skill) 
                        if G.nodes[pos]["type"]=="position" and pos != node]
            if neighbors:
                # Для каждого соседа вычисляем BM25 скор относительно пользовательского запроса
                neighbor_scores = []
                for neighbor_pos in neighbors:
                    neighbor_idx = G.nodes[neighbor_pos]["bm25_index"]
                    neighbor_score = bm25_scores[neighbor_idx]
                    neighbor_scores.append((neighbor_pos, neighbor_score))
                
                # Сортируем по BM25 скору и берем топ
                neighbor_scores.sort(key=lambda x: x[1], reverse=True)
                
                for neighbor_pos, _ in neighbor_scores[:top_career]:
                    career_paths.add(neighbor_pos)
                    
                    # Собираем навыки соседей
                    neighbor_skills = [
                        s for s in G.successors(neighbor_pos) if G.nodes[s]["type"] == "skill"
                    ]
                    all_neighbor_skills.extend(neighbor_skills)
    
    # Фильтрация навыков по частоте
    skill_counts = Counter(all_neighbor_skills)
    filtered_skills = [s for s, c in skill_counts.items() if c >= min_skill_freq]
    
    # Ограничиваем топ N по частоте
    expanded_skills = [s for s, _ in skill_counts.most_common(top_skills) if s in filtered_skills]

    print(f"Итого: {len(recommendations)} рекомендаций, {len(expanded_skills)} навыков, {len(career_paths)} карьерных путей")
    
    print("\n=== РЕКОМЕНДУЕМЫЕ ВАКАНСИИ ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Компания: {rec['company']}")
        print(f"   Опыт: {rec['experience']}")
        print(f"   Зарплата: {rec['salary']}")
        print(f"   Отрасль: {rec['industry']}")
        print(f"   Требования: {rec['requirements']}")
        print(f"   BM25 Score: {rec['bm25_score']:.3f}")
        print(f"   Навыки: {', '.join(rec['skills'][:5])}{'...' if len(rec['skills']) > 5 else ''}")
        print()
    
    print("=== РЕКОМЕНДУЕМЫЕ НАВЫКИ ДЛЯ РАЗВИТИЯ ===")
    for i, skill in enumerate(expanded_skills, 1):
        print(f"{i}. {skill}")
    
    print(f"\n=== ВОЗМОЖНЫЕ КАРЬЕРНЫЕ ПУТИ (топ-10) ===")
    for i, career in enumerate(list(career_paths)[:10], 1):
        print(f"{i}. {career}")
    
    return recommendations, expanded_skills, list(career_paths)

def get_relevant_vacancies_by_keywords(keywords, top_k=10):
    """
    Поиск вакансий по списку ключевых слов
    """
    # Объединяем ключевые слова в один запрос
    query = " ".join(keywords)
    user_tokens = tokenize_text(query)
    
    bm25_scores = bm25.get_scores(user_tokens)
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        node = position_nodes[idx]
        n_data = G.nodes[node]
        
        results.append({
            "title": node,
            "company": n_data["company"],
            "bm25_score": float(bm25_scores[idx]),
            "original_text": vacancy_texts[idx]
        })
    
    return results