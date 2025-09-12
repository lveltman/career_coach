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


QUESTION_BLOCKS = {
    'context': [
        "Привет! Расскажи, пожалуйста, какая у тебя сейчас должность и в какой сфере ты работаешь?",
        "Сколько лет у тебя общего опыта работы? А сколько именно в этой сфере/должности?",
        "Какие проекты ты считаешь самыми значимыми в своей работе?"
    ],
    'education': [
        "Какое у тебя образование? Расскажи про вуз, специальность или курсы, которые были важными.",
        "Какими профессиональными навыками ты владеешь лучше всего?",
        "Какие личные качества помогают тебе в работе?",
        "Какие языки программирования, инструменты или технологии ты чаще всего используешь?"
    ],
    'goals': [
        "Кем ты себя видишь через 1–3 года? Какая должность для тебя была бы следующей целью?",
        "Какой формат работы тебе ближе — офис, удалёнка или гибрид?",
        "Какой уровень дохода для тебя комфортный и мотивирующий?",
        "Что для тебя самое важное при выборе новой работы: стабильность, рост, интересные задачи, свобода, что-то ещё?"
    ]
}


# Системный промпт для валидации ответов
VALIDATION_PROMPT = """Ты — помощник карьерного коуча. Твоя задача — проверить, ответил ли пользователь на заданный вопрос достаточно информативно.

ВОПРОС: {question}
ОТВЕТ ПОЛЬЗОВАТЕЛЯ: {answer}

Критерии хорошего ответа:
- Ответ относится к заданному вопросу
- Содержит конкретную информацию, а не общие фразы
- Длина ответа больше 10 символов
- Не является отказом или уходом от темы

Ответь ТОЛЬКО "Да" или "Нет" без дополнительных пояснений."""


async def validate_answer(question: str, answer: str) -> bool:
    """Проверяет, подходит ли ответ пользователя к заданному вопросу"""
    if not answer or len(answer.strip()) < 5:
        return False
    
    validation_prompt = VALIDATION_PROMPT.format(question=question, answer=answer)
    messages = [{"role": "system", "content": validation_prompt}]
    
    try:
        llm_response = await wrapped_get_completion(
            MODEL_URL, API_TOKEN, messages, MODEL_NAME, 0.3, folder_id=config.FOLDER_ID
        )
        
        # Проверяем, содержит ли ответ "Да"
        return "да" in llm_response.lower().strip()[:10]
    
    except Exception as e:
        print(f"Ошибка валидации: {e}")
        # В случае ошибки считаем ответ валидным, чтобы не блокировать пользователя
        return True


def get_current_question(current_block: str, question_index: int) -> str:
    """Возвращает текущий вопрос для блока"""
    questions = QUESTION_BLOCKS.get(current_block, [])
    if question_index < len(questions):
        return questions[question_index]
    return None


def get_next_block_and_question(current_block: str, question_index: int):
    """Определяет следующий блок и вопрос"""
    questions = QUESTION_BLOCKS.get(current_block, [])
    
    # Если есть еще вопросы в текущем блоке
    if question_index + 1 < len(questions):
        return current_block, question_index + 1
    
    # Переход к следующему блоку
    block_order = list(QUESTION_BLOCKS.keys())
    current_block_index = block_order.index(current_block) if current_block in block_order else -1
    
    if current_block_index + 1 < len(block_order):
        next_block = block_order[current_block_index + 1]
        return next_block, 0
    
    # Все блоки пройдены
    return "recommendation", 0


async def chatbot_step(user_input, history, current_block, question_index, waiting_for_answer):
    
    # Если ждем ответ на конкретный вопрос
    if waiting_for_answer:
        current_question = get_current_question(current_block, question_index)
        
        if current_question:
            # Валидируем ответ
            is_valid = await validate_answer(current_question, user_input)
            
            if not is_valid:
                # Ответ не подходит, просим еще раз
                response = f"Пожалуйста, ответь более подробно на вопрос: {current_question}"
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                return history, current_block, question_index, True, response
            
            # Ответ подходит, сохраняем и переходим дальше
            history.append({"role": "user", "content": user_input})
            
            # Определяем следующий блок/вопрос
            next_block, next_question_index = get_next_block_and_question(current_block, question_index)
            
            if next_block == "recommendation":
                print("=" * 60)
                print("ВСЕ ОТВЕТЫ ПОЛЬЗОВАТЕЛЯ:")
                print("=" * 60)
                user_answers = [msg for msg in history if msg["role"] == "user"]
                for i, answer in enumerate(user_answers, 1):
                    print(f"{i}. {answer['content']}")
                print("=" * 60)

                career_goals = f"Сейчас я работаю: {user_answers[0]['content']}, через 1-3 года я бы хотел быть: {user_answers[7]['content']}"

                response = await generate_final_recommendations(history, career_goals)
                history.append({"role": "assistant", "content": response})
                return history, next_block, 0, False, response
            else:
                # Задаем следующий вопрос
                next_question = get_current_question(next_block, next_question_index)
                if next_question:
                    # Добавляем переходную фразу между блоками
                    if next_block != current_block:
                        if next_block == "education":
                            transition = "Отлично! Теперь расскажи про образование и навыки. "
                        elif next_block == "goals":
                            transition = "Понятно! Давай теперь поговорим о твоих карьерных целях. "
                        else:
                            transition = ""
                        response = transition + next_question
                    else:
                        response = next_question
                    
                    history.append({"role": "assistant", "content": response})
                    return history, next_block, next_question_index, True, response
    
    # Если не ждем ответ (начальное состояние или ошибка)
    first_question = get_current_question("context", 0)
    if first_question:
        history.append({"role": "assistant", "content": first_question})
        return history, "context", 0, True, first_question
    
    return history, current_block, question_index, waiting_for_answer, "Произошла ошибка. Попробуйте начать заново."


async def generate_final_recommendations(history, career_goals):
    """
    Улучшенная генерация финальных рекомендаций
    """
    
    # Собираем профиль из истории    
    user_profile_json, user_profile_text = process_user_profile_from_history(history)
    
    # Расширяем поисковый запрос контекстом из профиля
    enhanced_query = f"{career_goals}\n\nДополнительный контекст:\n{user_profile_text}"
    
    # Получаем рекомендации на основе расширенного профиля
    recommendations, expanded_skills, career_paths = recommend_vacancies(
        career_goals, 
        top_k=10, 
        top_career=2,
        min_skill_freq=2,
        top_skills=15
    )
    
    if not recommendations:
        return "К сожалению, не удалось найти подходящие рекомендации. Попробуйте уточнить ваши карьерные цели."
    
    # Улучшенный системный промпт
    # final_system_prompt = (
    #     "Ты — опытный карьерный коуч, специализирующийся на технических ролях в ML/AI. "
    #     "Твоя задача — проанализировать КОНКРЕТНЫЕ найденные вакансии и дать подробные (важно!) персональные рекомендации.\n\n"
        
    #     "ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:\n"
    #     "1. Используй ТОЛЬКО вакансии из списка 'found_positions' - не придумывай новые\n"
    #     "2. Упоминай конкретные компании и позиции по названиям\n"
    #     "3. Используй навыки из 'skills_to_develop' для планирования развития\n"
    #     "4. Рассматривай карьерные пути из 'career_paths'\n\n"
        
    #     "ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ:\n"
    #     "✅ 'Рекомендую позицию Senior NLP engineer в Just AI'\n"
    #     "✅ 'С Вашим опытом Вам подходит позиция ML Engineer в Сбере'\n"
    #     "❌ 'Рекомендую работу в крупной IT-компании' (слишком общее)\n"
    #     "❌ 'Советую найти позицию в Google' (компании нет в списке)\n\n"
        
    #     "ФОРМАТ ОТВЕТА (строго JSON):\n"
    #     "{\n"
    #     "  \"response\": \"Детальный анализ найденных вакансий с конкретными названиями компаний и позиций. Объясни почему именно эти вакансии подходят по целям пользователя. Обязательно упомяни релевантность.\",\n"
    #     "  \"recommendation\": {\n"
    #     "    \"nearest_position\": \"ТОЧНОЕ название позиции из found_positions с указанием компании\",\n"
    #     "    \"nearest_position_reason\": \"Почему именно эта позиция (компания + роль) наиболее подходящая сейчас, укажи релевантность\",\n"
    #     "    \"recommended_position\": \"Целевая позиция из found_positions для роста\",\n"
    #     "    \"recommended_position_reason\": \"Обоснование выбора с учетом карьерных целей\",\n"
    #     "    \"skills_gap\": \"Навыки из skills_to_develop, которые нужно развить для этих позиций\",\n"
    #     "    \"plan_1_2_years\": \"План развития на основе найденных требований из requirements, вакансий и навыков\",\n"
    #     "    \"recommended_courses\": [\"Курсы для развития навыков из skills_to_develop\"],\n"
    #     "    \"current_vacancies\": [\"Топ-3 самые подходящие вакансии с компаниями из found_positions\"]\n"
    #     "  }\n"
    #     "}\n\n"
        
    #     "КРИТИЧЕСКИ ВАЖНО: Не создавай новые вакансии - работай только с предоставленными данными!"
    # )
    final_system_prompt = (
        "Ты — опытный карьерный коуч, специализирующийся на технических ролях в ML/AI. "
        "Твоя задача — проанализировать КОНКРЕТНЫЕ найденные вакансии и дать подробные персональные рекомендации.\n\n"
        
        "ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:\n"
        "1. Используй ТОЛЬКО вакансии из списка 'found_positions' — не придумывай новые.\n"
        "2. Упоминай конкретные компании и позиции по названиям.\n"
        "3. Используй навыки из 'skills_to_develop' и требования из 'requirements' для плана развития.\n"
        "4. Рассматривай карьерные пути из 'career_paths'.\n"
        "5. Отвечай СТРОГО в формате JSON без лишнего текста.\n\n"
        
        "ПРИМЕР (few-shot):\n"
        "Входные данные:\n"
        "{\n"
        "  \"found_positions\": [\n"
        "    {\"title\": \"ML Engineer\", \"company\": \"Сбер\"},\n"
        "    {\"title\": \"Senior NLP Engineer\", \"company\": \"Just AI\"},\n"
        "    {\"title\": \"Data Scientist\", \"company\": \"Яндекс\"}\n"
        "  ],\n"
        "  \"skills_to_develop\": [\"NLP\", \"Deep Learning\"],\n"
        "  \"career_paths\": [\"ML Engineer → Senior ML Engineer\"]\n"
        "}\n\n"
        
        "Ожидаемый ответ:\n"
        "{\n"
        "  \"response\": \"С учётом вашего опыта оптимально начать с роли ML Engineer в Сбере — там требования совпадают с вашим профилем. "
        "Senior NLP Engineer в Just AI можно рассматривать как следующую цель, так как у вас есть базовые навыки NLP. "
        "Data Scientist в Яндексе также подходит, но менее релевантен.\",\n"
        "  \"recommendation\": {\n"
        "    \"nearest_position\": \"ML Engineer в Сбер\",\n"
        "    \"nearest_position_reason\": \"Эта роль наиболее близка к текущим навыкам, требования совпадают.\",\n"
        "    \"recommended_position\": \"Senior NLP Engineer в Just AI\",\n"
        "    \"recommended_position_reason\": \"Подходит для следующего карьерного шага, есть перспектива роста в NLP.\",\n"
        "    \"skills_gap\": \"Необходимо подтянуть NLP и Deep Learning.\",\n"
        "    \"plan_1_2_years\": \"В течение года укрепить экспертизу в NLP, через 2 года выйти на уровень Senior.\",\n"
        "    \"recommended_courses\": [\"Курс по NLP\", \"Advanced Deep Learning\"],\n"
        "    \"current_vacancies\": [\n"
        "      \"ML Engineer в Сбер\",\n"
        "      \"Senior NLP Engineer в Just AI\",\n"
        "      \"Data Scientist в Яндекс\"\n"
        "    ]\n"
        "  }\n"
        "}\n\n"
        
        # "КРИТИЧЕСКИ ВАЖНО: верни ответ ТОЛЬКО в JSON-формате как в примере выше. "
        # "Любой текст вне JSON — ошибка."
        "КРИТИЧЕСКИ ВАЖНО: Ответь строго JSON. Никакого текста вне JSON. Если нужно пояснение — включи его в поле response. Любой другой формат считается ошибкой»"
    )


    # Подготовка данных для модели
    payload = {
        "user_profile": user_profile_json,
        "user_goals": career_goals,
        "found_positions": [
            {
                "title": rec["title"],
                "company": rec["company"],
                "experience": rec["experience"],
                "salary": rec["salary"],
                "key_skills": rec["skills"][6:], 
                "requirements": rec["requirements"],
                "relevance_score": rec["similarity_score"]
            }
            for rec in recommendations[5:] 
        ],
        "skills_to_develop": expanded_skills,
        "career_paths": career_paths[:5],  # Топ-5 карьерных путей
    }

    user_message = f"""
АНАЛИЗИРУЙ СЛЕДУЮЩИЕ ДАННЫЕ И ДАЙ РЕКОМЕНДАЦИИ:

=== МОЙ ПРОФИЛЬ ===
{user_profile_json}

=== МОИ КАРЬЕРНЫЕ ЦЕЛИ ===
{career_goals}

=== НАЙДЕННЫЕ ДЛЯ МЕНЯ ВАКАНСИИ (ОБЯЗАТЕЛЬНО используй эти конкретные позиции) ===
{json.dumps(payload["found_positions"], ensure_ascii=False, indent=2)}

=== НАВЫКИ ДЛЯ РАЗВИТИЯ ===
{json.dumps(expanded_skills[:15], ensure_ascii=False)}

=== ВОЗМОЖНЫЕ КАРЬЕРНЫЕ ПУТИ ===
{json.dumps(career_paths[:8], ensure_ascii=False)}
"""
# ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
# "Рекомендую позицию 'Senior NLP engineer' в компании 'Just AI' как наиболее подходящую..."

# ЗАДАЧА: Проанализируй ЭТИ КОНКРЕТНЫЕ вакансии и выбери лучшие для пользователя. Упоминай названия компаний!

    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        llm_response = await wrapped_get_completion(
            MODEL_URL, API_TOKEN, messages, MODEL_NAME, MODEL_TEMP, folder_id=config.FOLDER_ID
        )
        print(f"[LLM reponse]: {llm_response}")
        # Улучшенное извлечение JSON
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                
                # Валидация результата
                if "response" in result and "recommendation" in result:
                    # res = result.get("response", "Рекомендации сформированы успешно!")
                    # res += "\nНАВЫКИ ДЛЯ РАЗВИТИЯ:\n"
                    # res += json.dumps(expanded_skills[:5], ensure_ascii=False)
                    # res += "ВОЗМОЖНЫЕ КАРЬЕРНЫЕ ПУТИ:\n"
                    # res += json.dumps(career_paths[:3], ensure_ascii=False)
                    res = parse_llm_response(result)
                    return res
                    # return result.get("response", "Рекомендации сформированы успешно!")
                else:
                    print(f"[ERROR] Неполный JSON ответ: {result}")
                    
            except json.JSONDecodeError as e:
                print(f"[ERROR] Ошибка парсинга JSON: {e}")
                print(f"[ERROR] Ответ LLM: {llm_response}")
        
        # Если JSON не извлечен, возвращаем как есть
        res = llm_response
        res += "\n\nНАВЫКИ ДЛЯ РАЗВИТИЯ:\n"
        res += json.dumps(expanded_skills[:5], ensure_ascii=False)
        res += "\n\nВОЗМОЖНЫЕ КАРЬЕРНЫЕ ПУТИ:\n"
        res += json.dumps(career_paths[:3], ensure_ascii=False)
        return llm_response
        
    except Exception as e:
        print(f"[ERROR] Ошибка при генерации рекомендаций: {e}")
        return f"Произошла ошибка при генерации рекомендаций: {e}"


def parse_llm_response(data: str) -> str:
    rec = data.get("recommendation", {})

    text = []
    text.append("🔎 Рекомендации по карьерным шагам:\n")

    text.append(f"{data.get('response')}\n")

    if rec.get("nearest_position"):
        text.append(f"📍 **Ближайшая позиция:** {rec['nearest_position']}")
        if rec.get("nearest_position_reason"):
            text.append(f"Причина: {rec['nearest_position_reason']}\n")

    if rec.get("recommended_position"):
        text.append(f"⭐ **Рекомендуемая следующая позиция:** {rec['recommended_position']}")
        if rec.get("recommended_position_reason"):
            text.append(f"Причина: {rec['recommended_position_reason']}\n")

    if rec.get("skills_gap"):
        text.append(f"🛠 Навыки, которые нужно развить: {rec['skills_gap']}\n")

    if rec.get("plan_1_2_years"):
        text.append(f"📅 План развития на 1–2 года:\n{rec['plan_1_2_years']}\n")

    if rec.get("recommended_courses"):
        courses = "\n".join([f"   • {c}" for c in rec['recommended_courses']])
        text.append(f"📚 Рекомендованные курсы:\n{courses}\n")

    if rec.get("current_vacancies"):
        vacancies = "\n".join([f"   • {v}" for v in rec['current_vacancies']])
        text.append(f"💼 Актуальные вакансии:\n{vacancies}")

    return "\n".join(text)

def sync_chatbot(user_input, history, current_block, question_index, waiting_for_answer):
    history, current_block, question_index, waiting_for_answer, response = asyncio.run(
        chatbot_step(user_input, history, current_block, question_index, waiting_for_answer)
    )
    return history, history, current_block, question_index, waiting_for_answer, ""


def reset_chat():
    first_question = get_current_question("context", 0)
    initial_history = [{"role": "assistant", "content": first_question}]
    return initial_history, initial_history, "context", 0, True, ""


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Career Coach")
    gr.Markdown("Отвечай на вопросы подробно, чтобы получить персональные карьерные рекомендации!")

    chatbot_ui = gr.Chatbot(
        value=[{"role": "assistant", "content": get_current_question("context", 0)}],
        type="messages"
    )

    msg = gr.Textbox(label="Ваш ответ:", placeholder="Введите ваш ответ здесь...")
    reset_btn = gr.Button("🔄 Начать заново")

    # Состояния
    history_state = gr.State(value=[{"role": "assistant", "content": get_current_question("context", 0)}])
    block_state = gr.State(value="context")
    question_index_state = gr.State(value=0)
    waiting_for_answer_state = gr.State(value=True)
    
    # Отправка сообщения
    msg.submit(
        sync_chatbot, 
        [msg, history_state, block_state, question_index_state, waiting_for_answer_state], 
        [chatbot_ui, history_state, block_state, question_index_state, waiting_for_answer_state, msg]
    )

    # Кнопка сброса
    reset_btn.click(
        reset_chat, 
        [], 
        [chatbot_ui, history_state, block_state, question_index_state, waiting_for_answer_state, msg]
    )


demo.launch()