import asyncio


# Mock Knowledge Graph вакансий
MOCK_VACANCIES = [
    {"title": "ML Engineer", "skills": ["Python", "PyTorch", "ML"], "track": "Data Science"},
    {"title": "Data Analyst", "skills": ["SQL", "Excel", "Visualization"], "track": "Data Analytics"},
    {"title": "Data Engineer", "skills": ["Spark", "ETL", "Python"], "track": "Data Engineering"},
]

# Простая RAG функция: выбирает вакансии с пересечением навыков
def rag_recommend(profile):
    user_skills = profile.get("skills", [])
    recommended = []
    for vac in MOCK_VACANCIES:
        overlap = set(user_skills) & set(vac["skills"])
        if overlap:
            recommended.append({**vac, "matched_skills": list(overlap)})
    return recommended