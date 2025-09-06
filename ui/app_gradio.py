import gradio as gr
import asyncio
import aiohttp
import tenacity
import json
import re

from services.model_api import wrapped_get_completion
from services.rag import rag_recommend

from config import config


API_URL = config.API_URL
API_TOKEN= config.API_TOKEN
MODEL_URL = config.MODEL_URL
MODEL_NAME = config.MODEL_NAME
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
    - текст для пользователя
    - следующий блок (или None, если блок не меняется)
    """
    try:
        data = extract_json(llm_output)
        response_text = data.get("response", "")
        next_block = data.get("current_block", None)
        return response_text, next_block
    except json.JSONDecodeError:
        return llm_output, None

def extract_json(llm_output: str):
    # ищем первый JSON-блок в тексте
    match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if not match:
        return {}
    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {}

async def chatbot_step(user_input, history, current_block):
    
    history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM_PROMPT_BLOCKS[current_block]}] + history

    llm_raw_output = await wrapped_get_completion(MODEL_URL, API_TOKEN, messages, MODEL_NAME, MODEL_TEMP)

    response, next_block = parse_llm_response(llm_raw_output)
        
    if next_block is not None:
        current_block = next_block

    # Логика перехода между блоками
    if "recommendation" in current_block:
        response += "\n\n👉 А теперь мои рекомендации."

        # Собираем профиль из истории
        mock_profile = {
            "skills": ["Python", "ML", "SQL"]  # пример
        }
        recs = rag_recommend(mock_profile)
        response += "\n\n🎯 Рекомендованные вакансии и треки:"
        for r in recs:
            response += f"\n- {r['title']} (Track: {r['track']}, matched skills: {', '.join(r['matched_skills'])})"

    print(f"fin response: {response}")

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    history.append({"role": "assistant", "content": response})

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
