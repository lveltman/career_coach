import re
import faiss
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
from sentence_transformers import SentenceTransformer


df_vacancies = pd.read_parquet('../data_artefacts')


model = SentenceTransformer('efederici/sentence-bert-base')

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

vacancy_texts = [
    normalize_text(
        f"{row['title']} {row['company']} {', '.join(row['skills'])} {row['experience']} {row['keywords']}"
    )
    for _, row in df_vacancies.iterrows()
]

vacancy_embeddings = model.encode(vacancy_texts, convert_to_numpy=True).astype("float32")
vacancy_embeddings /= np.linalg.norm(vacancy_embeddings, axis=1, keepdims=True)


# FAISS индекс
dim = vacancy_embeddings.shape[1]
faiss_index = faiss.IndexHNSWFlat(dim, 32)  # HNSW для быстрого поиска
faiss_index.add(vacancy_embeddings)


G = nx.DiGraph()
position_nodes = []

for i, row in df_vacancies.iterrows():
    pos_node = row["title"]
    position_nodes.append(pos_node)
    
    # Вершины вакансии
    G.add_node(pos_node, type="position", vacancy_id=row["vacancy_id"], 
               company=row["company"], experience=row["experience"],
               salary=row["salary_str"], industry=row["industry"],
               embedding=vacancy_embeddings[i])
    
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


def recommend_vacancies(user_text, top_k=5, top_career=1, min_skill_freq=2, top_skills=10):
    user_vector = model.encode([normalize_text(user_text)], convert_to_numpy=True).astype("float32")
    user_vector /= np.linalg.norm(user_vector, axis=1, keepdims=True)
    # Поиск топ-K вакансий по FAISS
    distances, indices = faiss_index.search(user_vector, top_k)
    
    recommendations = []
    career_paths = set()
    all_neighbor_skills = []
    
    for idx in indices[0]:
        node = position_nodes[idx]
        n_data = G.nodes[node]

        skills = [s for s in G.successors(node) if G.nodes[s]["type"]=="skill"]
        
        recommendations.append({
            "title": node,
            "company": n_data["company"],
            "experience": n_data["experience"],
            "salary": n_data["salary"],
            "industry": n_data["industry"],
            "skills": skills
        })
        
        # Топ-N похожих позиций через навыки
        for skill in skills:
            neighbors = [pos for pos in G.successors(skill) if G.nodes[pos]["type"]=="position" and pos != node]
            if neighbors:
                # Сортируем по косинусной близости эмбеды
                neighbor_embeddings = np.array([G.nodes[p]["embedding"] for p in neighbors]).astype("float32")
                sims = neighbor_embeddings @ user_vector.T
                # Берем только top_career ближайших соседей
                top_idx = np.argsort(-sims.ravel())[:top_career]
                for i in top_idx:
                    neighbor_pos = neighbors[i]
                    career_paths.add(neighbor_pos)
                    
                    # Вытаскиваем навыки у соседа
                    neighbor_skills = [
                        s for s in G.successors(neighbor_pos) if G.nodes[s]["type"] == "skill"
                    ]
                    all_neighbor_skills.extend(neighbor_skills)
    
    # Фильтрация навыков по частоте
    skill_counts = Counter(all_neighbor_skills)
    filtered_skills = [s for s, c in skill_counts.items() if c >= min_skill_freq]
    
    # Ограничиваем топ N по частоте
    expanded_skills = [s for s, _ in skill_counts.most_common(top_skills) if s in filtered_skills]

    return recommendations, expanded_skills, list(career_paths)

user_query = "ML engineer с опытом PyTorch 1 год"
recs, skills, paths = recommend_vacancies(user_query)

print("Рекомендованные вакансии (ближайшие к user vector через faiss:")
for r in recs:
    print(r)

print("\nНавыки для апгрейда (через граф):")
print(skills)

print("\nПотенциальные карьерные переходы:")
print(paths)