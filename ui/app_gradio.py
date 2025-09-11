import gradio as gr
import asyncio
import aiohttp
import tenacity
import json
import re

from services.model_api import wrapped_get_completion
from backend.rag import recommend_vacancies
from services.user_profile import process_user_profile_from_history

from config import config


API_TOKEN= config.API_TOKEN
MODEL_URL = config.MODEL_URL
MODEL_NAME = f"gpt://{config.FOLDER_ID}/{config.MODEL_NAME}" 
MODEL_TEMP = config.MODEL_TEMP
MAX_HISTORY = config.MAX_HISTORY


SYSTEM_PROMPT_BLOCKS = {
    "context": """Ты — карьерный коуч. Сначала собери стартовый контекст:
- профессиональная сфера
- текущая должность или позиция
- опыт работы в годах
- реализованные проекты

Задавай вопросы по одному и дружелюбно общайся с пользователем. 
В конце блока (когда задашь вопросы по всем пунктам) ответь **только в JSON** формате, без дополнительного текста (важно!):

{
  "response": "Текст, который нужно показать пользователю",
  "current_block": "goals"  # когда блок завершен, иначе оставляй "context"
}""",

    "goals": """Ты — карьерный коуч. Теперь уточни цели:
- интересующая сфера и специализация
- какие активности и функции привлекают
- амбиции по должности и зарплате

Задавай вопросы по одному, используя данные из блока context.

В конце блока (когда задашь вопросы по всем пунктам) ответь **только в JSON** формате, без дополнительного текста (важно!):
{
  "response": "Текст для пользователя",
  "current_block": "skills"  # когда блок завершен
}""",

    "skills": """Ты — карьерный коуч. Собери данные о навыках:
- hard skills и инструменты
- soft skills
- образование и курсы

Задавай уточняющие вопросы дружелюбно. 
В конце блока (когда задашь вопросы по всем пунктам) ответь **только в JSON** формате, без дополнительного текста (важно!):

{
  "response": "Текст для пользователя",
  "current_block": "recommendation"  # когда блок завершен
}""",

    "recommendation": """Ты — карьерный коуч. На основе всех предыдущих блоков дай рекомендации. 
В конце блока (когда задашь вопросы по всем пунктам) ответь **только в JSON** формате, без дополнительного текста (важно!):

{
  "response": "Объяснение и рекомендации для пользователя",
  "current_block": recommendation,
  "recommendation": {
      "nearest_position": "",
      "nearest_position_reason": "",
      "recommended_position": "",
      "recommended_position_reason": "",
      "skills_gap": "",
      "plan_1_2_years": "",
      "recommended_courses": [],
      "current_vacancies": []
  }
}""",

    "rag": "Подбери вакансии и треки развития на основе профиля пользователя"
}


def parse_llm_response(llm_output):
    """
    Принимает строку от LLM и возвращает:
    - текст для пользователя (fallback к «сырому» тексту, если JSON пуст)
    - следующий блок (или None, если блок не меняется)
    """
    # 1) Пытаемся вытащить JSON-объект
    data = extract_json(llm_output)
    response_text = ""
    next_block = None

    if data:
        response_text = (data.get("response") or "").strip()
        next_block = data.get("current_block", None)

    # 2) Если JSON есть, но response пустой — берём текст до JSON как fallback
    if not response_text:
        # Убираем код-блоки ```...``` и берём префикс до первого { (если есть)
        text = llm_output
        # отрежем все блоки ```...```
        text = re.sub(r"```[\\s\\S]*?```", "", text)
        # префикс до JSON
        prefix = text.split("{", 1)[0].strip()
        # если префикс пуст, вернём весь текст без код-блоков
        response_text = prefix or text.strip()

    return response_text, next_block

def extract_json(llm_output: str):
    # ищем первый JSON-блок по самой простой скобочной эвристике
    match = re.search(r'\\{[\\s\\S]*\\}', llm_output)
    if not match:
        return {}
    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # иногда попадает хвост после JSON — попробуем «почистить» хвостовые запятые/комментарии,
        # но по-минимуму просто вернём пусто, чтобы сработал fallback
        return {}

async def chatbot_step(user_input, history, current_block):
    
    history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM_PROMPT_BLOCKS[current_block]}] + history

    llm_raw_output = await wrapped_get_completion(MODEL_URL, API_TOKEN, messages, MODEL_NAME, MODEL_TEMP, folder_id=config.FOLDER_ID)
    print("[LLM][RAW_INIT]", llm_raw_output)

    response, next_block = parse_llm_response(llm_raw_output)
        
    if next_block is not None:
        current_block = next_block

    print(f"fin response: {response}")

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    history.append({"role": "assistant", "content": response})
    
    # Логика перехода между блоками
    if "recommendation" in current_block:
        # Собираем профиль из истории    
        user_profile_json, user_profile_text = process_user_profile_from_history(history)
        
        # Получаем рекомендации на основе профиля пользователя
        recommendations, expanded_skills, career_paths = recommend_vacancies(user_profile_text)
        # Формируем строгий финальный промпт и передаем компактные данные
        final_system_prompt = (
            "Ты — карьерный коуч. Сформируй финальные рекомендации на основе данных пользователя, "
            "подобранных вакансий, навыков для апгрейда и потенциальных карьерных переходов. "
            "Отвечай строго ОДНИМ JSON без какого-либо дополнительного текста. Структура ответа:\n\n"
            "{\n"
            "  \"response\": \"Краткое объяснение рекомендаций\",\n"
            "  \"current_block\": \"recommendation\",\n"
            "  \"recommendation\": {\n"
            "    \"nearest_position\": \"\",\n"
            "    \"nearest_position_reason\": \"\",\n"
            "    \"recommended_position\": \"\",\n"
            "    \"recommended_position_reason\": \"\",\n"
            "    \"skills_gap\": \"\",\n"
            "    \"plan_1_2_years\": \"\",\n"
            "    \"recommended_courses\": [],\n"
            "    \"current_vacancies\": []\n"
            "  }\n"
            "}"
        )

        # Собираем компактный пользовательский ввод для модели
        compact_payload = {
            "user_profile": user_profile_json,
            "recommendations": recommendations,
            "skills_to_develop": expanded_skills,
            "career_paths": career_paths,
        }

        final_messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": json.dumps(compact_payload, ensure_ascii=False)},
        ]

        llm_raw_output = await wrapped_get_completion(MODEL_URL, API_TOKEN, messages, MODEL_NAME, MODEL_TEMP, folder_id=config.FOLDER_ID)
        print("[LLM][RAW_FINAL]", llm_raw_output)

        # Парсим полный JSON-ответ и сохраняем его в историю как последний элемент
        final_json = extract_json(llm_raw_output)
        # Если парсинг не удался, хотя бы вернем сырое содержимое
        content_for_history = json.dumps(final_json, ensure_ascii=False) if final_json else llm_raw_output
        history[-1] = {"role": "assistant", "content": content_for_history}
        # Для совместимости вернем тот же контент наружу
        response = content_for_history


    return history, current_block, response


def sync_chatbot(user_input, history, current_block):
    history, current_block, response = asyncio.run(chatbot_step(user_input, history, current_block))
    return history, history, current_block, ""


def reset_chat():
    initial_history = [
        {"role": "assistant", "content": "Привет! Давай начнём. Расскажи, пожалуйста, в какой профессиональной сфере ты сейчас работаешь?"}
    ]
    return initial_history, initial_history, "context", ""



with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Career Coach")

    chatbot_ui = gr.Chatbot(value=[
        {"role": "assistant", "content": "Привет! Давай начнём. Расскажи, пожалуйста, в какой профессиональной сфере ты сейчас работаешь?"}
    ], type="messages")

    msg = gr.Textbox(label="Ваш ответ:")
    reset_btn = gr.Button("🔄 Начать заново")

    history_state = gr.State(value=[
        {"role": "assistant", "content": "Привет! Давай начнём. Расскажи, пожалуйста, в какой профессиональной сфере ты сейчас работаешь?"}
    ])
    block_state = gr.State(value="context")
    
    # Отправка сообщения
    msg.submit(sync_chatbot, [msg, history_state, block_state], [chatbot_ui, history_state, block_state, msg])

    # Кнопка сброса
    reset_btn.click(reset_chat, [], [chatbot_ui, history_state, block_state, msg])


demo.launch()