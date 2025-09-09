import polars as pl
from schema import CandidateProfile
from vectorize import VacancySearchEngine

# Загрузка данных
df = pl.read_parquet("./vacancies_final.parquet")[:100]
print("read")
# Инициализация и обучение модели
search_engine = VacancySearchEngine("efederici/sentence-bert-base")
search_engine.fit(df)
print("fit")
# Поиск вакансий
results = search_engine.search(
    "Data Scientist анализ данных машинное обучение Python",
    top_n=5,

)

# Вывод результатов
print(results.select(["vacancy_id", "title", "company", "similarity_score"]))