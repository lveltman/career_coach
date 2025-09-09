from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

# Определение модели CandidateProfile (как в предыдущем примере)
class ExperienceLevel(str, Enum):
    NO_EXPERIENCE = "нет опыта"
    ONE_TO_THREE = "от 1 до 3 лет"
    THREE_TO_SIX = "от 3 до 6 лет"
    MORE_THAN_SIX = "более 6 лет"

class CandidateProfile(BaseModel):
    """Модель для представления профиля кандидата."""
    
    requirement_responsibility: str = Field(
        ...,
        description="Сырой текст от пользователя о своих навыках и опыте"
    )
    
    skills: List[str] = Field(
        default=[],
        description="Список нормализованных навыков"
    )
    
    experience: ExperienceLevel = Field(
        default=ExperienceLevel.NO_EXPERIENCE,
        description="Категория опыта работы"
    )
    
    def to_bert_string(self) -> str:
        """Преобразует профиль в строку для векторизации."""
        parts = []
        parts.append(f"Опыт и обязанности: {self.requirement_responsibility}")
        
        if self.skills:
            skills_text = ", ".join(self.skills)
            parts.append(f"Навыки: {skills_text}")
        
        parts.append(f"Опыт: {self.experience.value}")
        
        return ". ".join(parts)
