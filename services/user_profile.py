"""
Модуль для формирования профиля пользователя на основе данных из блоков context, goals и skills.
"""
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class UserProfile:
    """Структура профиля пользователя."""
    title: str
    company: str
    skills: List[str]
    experience: str
    keywords: str  # или другое поле вместо keywords из вакансий


def extract_user_data_from_history(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Извлекает данные пользователя из истории чата.
    
    Args:
        history: История сообщений чата
        
    Returns:
        Словарь с извлеченными данными пользователя
    """
    user_data = {
        "context": {},
        "goals": {},
        "skills": {}
    }
    
    # Собираем все сообщения пользователя
    user_messages = []
    for message in history:
        if message.get("role") == "user":
            user_messages.append(message.get("content", ""))
    
    # Объединяем все сообщения пользователя в один текст
    full_user_text = " ".join(user_messages)
    
    # Извлекаем информацию с помощью простых паттернов
    # Это можно улучшить с помощью более сложной NLP обработки
    
    # Извлекаем профессиональную сферу
    context_patterns = {
        "professional_field": r"(?:работаю|специализируюсь|занимаюсь|сфера|область).*?(?:в|по|на)\s+([^.!?]+)",
        "current_position": r"(?:должность|позиция|я)\s+([^.!?]+?)(?:\s|$|[.!?])",
        "company": r"(?:в компании|работаю в|компания)\s+([А-Яа-яA-Za-z0-9\s]+?)(?:\s|$|[.!?])",
        "company_yandex": r"(Яндекс|Yandex)",
        "company_google": r"(Google|Гугл)",
        "company_microsoft": r"(Microsoft|Майкрософт)",
        "experience_years": r"(\d+)\s*(?:лет|года|год)\s*(?:опыта|работы)",
        "projects": r"(?:проект|реализовал|делал|участвовал).*?([^.!?]+)"
    }
    
    for key, pattern in context_patterns.items():
        match = re.search(pattern, full_user_text, re.IGNORECASE)
        if match:
            user_data["context"][key] = match.group(1).strip()
    
    # Извлекаем цели
    goals_patterns = {
        "target_field": r"(?:интересуюсь|хочу|цель|планирую).*?(?:сфера|область|направление).*?([^.!?]+)",
        "activities": r"(?:активности|функции|задачи|хочу).*?([^.!?]+)",
        "ambitions": r"(?:амбиции|цель|хочу|планирую).*?(?:должность|зарплата|позиция).*?([^.!?]+)"
    }
    
    for key, pattern in goals_patterns.items():
        match = re.search(pattern, full_user_text, re.IGNORECASE)
        if match:
            user_data["goals"][key] = match.group(1).strip()
    
    # Извлекаем навыки
    skills_patterns = {
        "hard_skills": r"(?:навыки|умею|знаю|владею|инструменты).*?([^.!?]+?)(?:\s|$|[.!?])",
        "soft_skills": r"(?:soft skills|мягкие навыки|личные качества).*?([^.!?]+?)(?:\s|$|[.!?])",
        "education": r"(?:образование|курсы|обучение|изучал).*?([^.!?]+?)(?:\s|$|[.!?])"
    }
    
    for key, pattern in skills_patterns.items():
        match = re.search(pattern, full_user_text, re.IGNORECASE)
        if match:
            user_data["skills"][key] = match.group(1).strip()
    
    return user_data


def create_user_profile_json(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создает JSON профиль пользователя в формате, аналогичном вакансиям.
    
    Args:
        user_data: Извлеченные данные пользователя
        
    Returns:
        JSON профиль пользователя
    """
    # Формируем title на основе текущей позиции или профессиональной сферы
    current_position = user_data["context"].get("current_position", "")
    professional_field = user_data["context"].get("professional_field", "")
    
    if current_position:
        title = current_position
    elif professional_field:
        title = professional_field
    else:
        title = "Специалист"
    
    # Формируем company
    company = user_data["context"].get("company", "")
    
    # Проверяем специальные случаи компаний
    if user_data["context"].get("company_yandex"):
        company = "Яндекс"
    elif user_data["context"].get("company_google"):
        company = "Google"
    elif user_data["context"].get("company_microsoft"):
        company = "Microsoft"
    
    # Собираем все навыки
    skills = []
    
    # Hard skills
    hard_skills_text = user_data["skills"].get("hard_skills", "")
    if hard_skills_text:
        # Простое разделение по запятым и точкам
        hard_skills = [skill.strip() for skill in re.split(r'[,;.]', hard_skills_text) if skill.strip()]
        # Фильтруем слишком длинные "навыки" (вероятно, это не навыки)
        hard_skills = [skill for skill in hard_skills if len(skill) < 50]
        skills.extend(hard_skills)
    
    # Soft skills
    soft_skills_text = user_data["skills"].get("soft_skills", "")
    if soft_skills_text:
        soft_skills = [skill.strip() for skill in re.split(r'[,;.]', soft_skills_text) if skill.strip()]
        skills.extend(soft_skills)
    
    # Опыт работы
    experience_years = user_data["context"].get("experience_years", "")
    if experience_years:
        try:
            years = int(re.search(r'\d+', experience_years).group())
            if years == 0:
                experience = "нет опыта"
            elif years <= 3:
                experience = "от 1 года до 3 лет"
            elif years <= 6:
                experience = "от 3 до 6 лет"
            else:
                experience = "более 6 лет"
        except:
            experience = "опыт не указан"
    else:
        experience = "опыт не указан"
    
    # Формируем keywords (вместо keywords из вакансий используем цели и интересы)
    keywords_parts = []
    
    # Добавляем целевые области
    target_field = user_data["goals"].get("target_field", "")
    if target_field:
        keywords_parts.append(target_field)
    
    # Добавляем интересующие активности
    activities = user_data["goals"].get("activities", "")
    if activities:
        keywords_parts.append(activities)
    
    # Добавляем амбиции
    ambitions = user_data["goals"].get("ambitions", "")
    if ambitions:
        keywords_parts.append(ambitions)
    
    # Добавляем проекты
    projects = user_data["context"].get("projects", "")
    if projects:
        keywords_parts.append(projects)
    
    keywords = " ".join(keywords_parts)
    
    return {
        "title": title,
        "company": company,
        "skills": skills,
        "experience": experience,
        "keywords": keywords
    }


def create_user_profile_text(user_profile_json: Dict[str, Any]) -> str:
    """
    Создает текст профиля пользователя для FAISS поиска.
    
    Args:
        user_profile_json: JSON профиль пользователя
        
    Returns:
        Текст профиля для векторизации
    """
    # Формируем текст аналогично тому, как это делается для вакансий в rag.py
    # f"{row['title']} {row['company']} {', '.join(row['skills'])} {row['experience']} {row['keywords']}"
    
    skills_text = ", ".join(user_profile_json.get("skills", []))
    
    profile_text = f"{user_profile_json.get('title', '')} {user_profile_json.get('company', '')} {skills_text} {user_profile_json.get('experience', '')} {user_profile_json.get('keywords', '')}"
    
    return profile_text.strip()


def process_user_profile_from_history(history: List[Dict[str, str]]) -> tuple[Dict[str, Any], str]:
    """
    Полный процесс обработки истории чата для создания профиля пользователя.
    
    Args:
        history: История сообщений чата
        
    Returns:
        Tuple с JSON профилем и текстом профиля для FAISS
    """
    # Извлекаем данные из истории
    user_data = extract_user_data_from_history(history)
    
    # Создаем JSON профиль
    user_profile_json = create_user_profile_json(user_data)
    
    # Создаем текст профиля для FAISS
    user_profile_text = create_user_profile_text(user_profile_json)
    
    return user_profile_json, user_profile_text