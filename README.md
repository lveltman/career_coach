# Карьерный коуч

## Описание

career_coach — это интерактивный сервис карьерного коучинга для технических специалистов, использующий ML/AI для анализа профиля пользователя и рекомендаций по вакансиям. Система собирает информацию о вашем опыте, навыках и целях, а затем подбирает релевантные вакансии и карьерные пути на основе современных методов поиска и графового анализа.

## Структура проекта

- `ui/` — Gradio-интерфейс для диалога с пользователем ([ui/app_gradio.py](ui/app_gradio.py))
- `services/` — обработка профиля пользователя и взаимодействие с LLM ([services/user_profile.py](services/user_profile.py), [services/model_api.py](services/model_api.py))
- `backend/` — поиск и рекомендации вакансий на основе BM25 и графа ([backend/rag.py](backend/rag.py))
- `vectorize/` — векторизация вакансий и профилей кандидатов через Sentence-BERT и FAISS ([vectorize/vectorize.py](vectorize/vectorize.py), [vectorize/schema.py](vectorize/schema.py))
- `parser/` — парсер вакансий с hh.ru ([parser/vacancy_parser.py](parser/vacancy_parser.py))
- `data_artefacts/` — артефакты данных, включая датасет вакансий в формате parquet

## Быстрый старт

1. **Настройте переменные окружения**  
   Создайте файл `.env` в корне проекта и пропишите:
   ```
    API_TOKEN=""
    MODEL_URL=""
    MODEL_NAME=""
    MODEL_TEMP=""
    MAX_HISTORY=""
   ```

2. **Установите зависимости**
   ```sh
   pip install -r requirements.txt
   ```

3. **Запустите Gradio-интерфейс**
   ```sh
   python3 -m ui.app_gradio
   ```

4. **(Опционально) Соберите датасет вакансий**
   - Используйте [parser/vacancy_parser.py](parser/vacancy_parser.py) для парсинга вакансий с hh.ru.
   - Данные сохраняются в [data_artefacts/vacancy_final.parquet](data_artefacts/vacancy_final.parquet).

5. **(Опционально) Постройте FAISS-индекс**
   - Пример: [vectorize/example.py](vectorize/example.py)

## Основные компоненты

- **Gradio UI**: диалоговый интерфейс, пошагово собирающий информацию о пользователе.
- **LLM API**: валидация и генерация рекомендаций через YandexGPT или совместимую модель ([services/model_api.py](services/model_api.py)).
- **RAG backend**: поиск вакансий по BM25 и графовый анализ навыков ([backend/rag.py](backend/rag.py)).
- **Vector Search**: поиск по эмбеддингам через Sentence-BERT и FAISS ([vectorize/vectorize.py](vectorize/vectorize.py)).
- **Парсер вакансий**: сбор вакансий с hh.ru ([parser/vacancy_parser.py](parser/vacancy_parser.py)).

## Пример запуска

```sh
python3 -m ui.app_gradio
```

## Лицензия

Проект распространяется под лицензией Apache 2.0. См. [LICENSE](LICENSE).

## Контакты

Для вопросов и предложений — создайте issue или pull request.
