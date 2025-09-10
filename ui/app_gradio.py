import gradio as gr
import asyncio
import aiohttp
import tenacity
import json
import re

from services.model_api import wrapped_get_completion
from backend.rag import rag_recommend

from config import config


API_TOKEN= config.API_TOKEN
MODEL_URL = config.MODEL_URL
# MODEL_NAME = f"gpt://{config.FOLDER_ID}/{config.MODEL_NAME}" # for deepseek: config.MODEL_NAME
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

    print(f"fin response: {response}")

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    history.append({"role": "assistant", "content": response})
    
    # Логика перехода между блоками
    if "recommendation" in current_block:
        # Собираем профиль из истории    
        # Тут нужно, чтобы ллм собрала json подобный ответ про юзера в формате, чтобы мы подали этот текст в эмбеддер
        user_profile_text = f"{user_profile['title']} {user_profile['company']} {', '.join(user_profile['skills'])} {user_profile['experience']} {row['keywords']}"
        # keywords - это типа требования в вакансиях, вместо этого можно от юзера другую инфу сюда вписывать
        recommendations, expanded_skills, career_paths = recommend_vacancies(user_profile_text)
        # надо взять recommendations, expanded_skills, career_paths и user_profile_text оформить в промпт и отправить ллм (вызвать еще раз думаю тут надо)
        # создаем тут промпт еще один последний        
        rec_prompt = f"""Ты — карьерный коуч. На основе всех предыдущих блоков и вот этих данных по вакансиям для пользователя: {recommendations}, собранным навыкам для апгрейда: {expanded_skills} и потенциальные карьерные переходам: {career_paths} дай рекомендации. 
    В конце блока ответь **только в JSON** формате, без дополнительного текста (важно!):
    
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
    }"""
        
        messages = [{"role": "system", "content": rec_prompt}] + history
        # response += "\n\n👉 А теперь мои рекомендации."

        llm_raw_output = await wrapped_get_completion(MODEL_URL, API_TOKEN, messages, MODEL_NAME, MODEL_TEMP)

        response, next_block = parse_llm_response(llm_raw_output)
        # тут достанутся только ее объяснения в response, по идее на основе промпта она должна
        # вернуть еще поле "recommendation" и его можно распарсить и вернуть юзеру только его или объяснение, хз, надо тестить
        history[-1] = {"role": "assistant", "content": response}


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
