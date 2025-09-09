import polars as pl
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from schema import CandidateProfile, ExperienceLevel



class VacancySearchEngine:
    """Класс для поиска вакансий с использованием Sentence-BERT и FAISS."""
    
    def __init__(self, model_name: str = "efederici/sentence-bert-base"):
        """
        Инициализация поискового движка.
        
        Args:
            model_name: Название модели из Hugging Face Hub
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.df = None
        self.dimension = None
        self.vacancy_profiles = []
    
    def _create_vacancy_profile(self, row: dict) -> CandidateProfile:
        """Создает CandidateProfile из строки вакансии."""
        # Формируем requirement_responsibility
        req_resp = ""
        if "requirement" in row and row["requirement"]:
            req_resp += row["requirement"]
        if "responsibility" in row and row["responsibility"]:
            if req_resp:
                req_resp += " "
            req_resp += row["responsibility"]
        
        # Извлекаем навыки
        skills = []
        if "skills" in row and row["skills"]:
            if isinstance(row["skills"], str):
                # Предполагаем, что навыки разделены запятыми
                skills = [skill.strip() for skill in row["skills"].split(",")]
            elif isinstance(row["skills"], list):
                skills = row["skills"]
        
        # Определяем уровень опыта
        experience = ExperienceLevel.NO_EXPERIENCE
        if "experience" in row and row["experience"]:
            exp_str = row["experience"].lower()
            if "1" in exp_str and "3" in exp_str:
                experience = ExperienceLevel.ONE_TO_THREE
            elif "3" in exp_str and "6" in exp_str:
                experience = ExperienceLevel.THREE_TO_SIX
            elif "6" in exp_str or "более" in exp_str:
                experience = ExperienceLevel.MORE_THAN_SIX
        
        return CandidateProfile(
            requirement_responsibility=req_resp,
            skills=skills,
            experience=experience
        )
    
    def fit(self, df: pl.DataFrame) -> None:
        """
        Векторизация вакансий и создание FAISS индекса.
        
        Args:
            df: DataFrame Polars с вакансиями
        """
        self.df = df
        
        # Создаем профили вакансий
        self.vacancy_profiles = []
        texts = []
        
        for row in df.iter_rows(named=True):
            profile = self._create_vacancy_profile(row)
            self.vacancy_profiles.append(profile)
            texts.append(profile.to_bert_string())
        
        # Векторизуем тексты
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self.dimension = embeddings.shape[1]
        
        # Создаем и заполняем FAISS индекс
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Индекс создан для {len(embeddings)} вакансий, размерность: {self.dimension}")
    
    def search(self, query: Union[str, CandidateProfile], top_n: int = 5, filters: Dict[str, Any] = None) -> pl.DataFrame:
        """
        Поиск вакансий по текстовому запросу или профилю кандидата.
        
        Args:
            query: Текстовый запрос или объект CandidateProfile
            top_n: Количество возвращаемых результатов
            filters: Словарь фильтров (поле: значение)
            
        Returns:
            DataFrame с результатами поиска
        """
        if self.index is None or self.df is None:
            raise ValueError("Сначала необходимо обучить модель методом fit()")
        
        # Преобразуем запрос в строку для векторизации
        if isinstance(query, CandidateProfile):
            query_text = query.to_bert_string()
        else:
            query_text = query
        
        # Векторизуем запрос
        query_vector = self.model.encode([query_text], convert_to_numpy=True).astype(np.float32)
        
        # Ищем ближайшие вакансии
        distances, indices = self.index.search(query_vector, min(top_n * 3, len(self.df)))
        print(query_vector)
        # Создаем DataFrame с результатами
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.df):  # Проверка на валидность индекса
                continue
                
            row = self.df.row(idx, named=True)
            
            # Применяем фильтры, если они заданы
            if filters:
                skip = False
                for field, value in filters.items():
                    if field in row and row[field] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Добавляем результат
            result = dict(row)
            result["similarity_score"] = float(1 / (1 + distances[0][i]))  # Преобразуем расстояние в оценку
            results.append(result)
            
            if len(results) >= top_n:
                break
        
        return pl.DataFrame(results)
    
    def search_by_profile(self, profile: CandidateProfile, top_n: int = 5, filters: Dict[str, Any] = None) -> pl.DataFrame:
        """
        Поиск вакансий по профилю кандидата.
        
        Args:
            profile: Профиль кандидата
            top_n: Количество возвращаемых результатов
            filters: Словарь фильтров (поле: значение)
            
        Returns:
            DataFrame с результатами поиска
        """
        return self.search(profile, top_n, filters)
    
    def save_index(self, index_path: str) -> None:
        """Сохраняет FAISS индекс на диск."""
        if self.index is None:
            raise ValueError("Индекс не создан")
        faiss.write_index(self.index, index_path)
    
    def load_index(self, index_path: str, df: pl.DataFrame) -> None:
        """Загружает FAISS индекс с диска."""
        self.df = df
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        
        # Восстанавливаем профили вакансий
        self.vacancy_profiles = []
        for row in df.iter_rows(named=True):
            self.vacancy_profiles.append(self._create_vacancy_profile(row))