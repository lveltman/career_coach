import asyncio
import aiohttp
import tenacity
import json
import os
import re

DEBUG = os.getenv("DEBUG_LLM", "0") == "1"

class ModelAPIError(Exception):
    pass

def normalize_model_uri(uri: str) -> str:
    if uri.startswith("gpt://") and not uri.endswith("/latest"):
        return uri.rstrip("/") + "/latest"
    return uri

def build_model_uri(model: str, folder_id: str) -> str:
    if model.startswith("gpt://"):
        return normalize_model_uri(model)
    if not folder_id:
        return model
    return f"gpt://{folder_id}/{model}/latest"

def to_yandex_messages(messages):
    """Конвертация сообщений в формат YandexGPT"""
    out = []
    for m in messages:
        text = (m.get("text") or m.get("content") or "").strip()
        if not text:
            continue
        role = m.get("role", "user")
        # YandexGPT поддерживает: system, user, assistant
        if role not in ["system", "user", "assistant"]:
            role = "user"
        out.append({"role": role, "text": text})
    return out

def clean_yandex_hallucination(text: str) -> str:
    """
    Специальная очистка для YandexGPT галлюцинаций
    Обрезает ответ до первых \n\nПользователь или \n\nАссистент
    """
    if not text or not isinstance(text, str):
        return ""
    
    # ГЛАВНОЕ: обрезаем до первого появления галлюцинированного диалога
    cut_patterns = [
        r'\n\nПользователь',
        r'\n\nАссистент'
    ]
    
    min_cut_pos = len(text)
    for pattern in cut_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            min_cut_pos = min(min_cut_pos, match.start())
    
    # Обрезаем текст
    if min_cut_pos < len(text):
        text = text[:min_cut_pos].strip()
    
    # Если после обрезки остался только текст без JSON, пытаемся найти JSON
    json_patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # JSON в блоке кода
        r'(\{[^{}]*"(?:response|current_block)"[^{}]*\})',  # Простой JSON
        r'(\{(?:[^{}]|{[^{}]*})*\})'  # Любой JSON объект
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                # Проверяем что это валидный JSON
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue
    
    # Если JSON не найден, возвращаем обрезанный текст
    return text.strip()

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    retry=tenacity.retry_if_exception_type((aiohttp.ClientError, ModelAPIError))
)
async def get_completion(url, token, messages, model, temperature=0.7, folder_id: str = ""):
    model_uri = build_model_uri(model, folder_id)
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {token}",
        "x-folder-id": folder_id 
    }
    
    processed_messages = to_yandex_messages(messages)
    
    # Добавляем строгий системный промпт для YandexGPT
    has_system = any(msg.get("role") == "system" for msg in processed_messages)
    if not has_system:
        system_prompt = {
            "role": "system", 
            "text": "Ты должен отвечать ТОЛЬКО в заданном формате. Не генерируй диалоги. Не добавляй 'Пользователь:' или 'Ассистент:'. Не задавай вопросы. Дай только запрошенный ответ."
        }
        processed_messages.insert(0, system_prompt)
    
    payload = {
        "modelUri": model_uri,
        "messages": processed_messages,
        "completionOptions": {
            "stream": False,
            "temperature": min(temperature, 0.1), 
            "maxTokens": "1024", 
            "reasoningOptions": {"mode": "DISABLED"},
        },
    }
    
    if "yandexgpt" in model.lower():
        payload["completionOptions"]["stopSequences"] = [
            "Пользователь:",
            "Ассистент:", 
            "\n\nПользователь",
            "\n\nАссистент"
        ]
    
    timeout = aiohttp.ClientTimeout(total=60)
    
    if DEBUG:
        print("[LLM][YandexGPT][REQ]", json.dumps(payload, ensure_ascii=False, indent=2))
    
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async with session.post(url=url, json=payload) as resp:
            text = await resp.text()
            
            if DEBUG:
                print(f"[LLM][YandexGPT][HTTP] {resp.status}")
                print(f"[LLM][YandexGPT][RAW] {text[:2000]}")
            
            if resp.status != 200:
                raise ModelAPIError(f"YandexGPT HTTP {resp.status}: {text}")

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                raise ModelAPIError(f"Invalid JSON from YandexGPT: {text[:500]}")
            
            if DEBUG:
                print("[LLM][YandexGPT][PARSED]", data)

            # Обработка YandexGPT формата
            if "result" in data:
                alts = data["result"].get("alternatives", [])
                if alts and "message" in alts[0]:
                    raw_text = alts[0]["message"].get("text", "").strip()
                    
                    if not raw_text:
                        return '{"response": "", "error": "empty_response"}'
                    
                    # Специальная очистка для YandexGPT
                    cleaned_text = clean_yandex_hallucination(raw_text)
                    
                    if DEBUG:
                        print(f"[LLM][YandexGPT][CLEANED] {cleaned_text}")
                    
                    return cleaned_text if cleaned_text else '{"response": "", "error": "hallucination_cleaned"}'
                
                # Если нет alternatives, возвращаем как есть
                return json.dumps(data["result"], ensure_ascii=False)

            # OpenAI-совместимый формат (на всякий случай)
            if "choices" in data:
                raw_text = data["choices"][0]["message"]["content"]
                return clean_yandex_hallucination(raw_text)

            raise ModelAPIError(f"Unexpected YandexGPT schema: {json.dumps(data, ensure_ascii=False)[:1000]}")

async def wrapped_get_completion(*args, **kwargs):
    try:
        return await get_completion(*args, **kwargs)
    except Exception as e:
        error_msg = f"[Error YandexGPT]: {str(e)}"
        if DEBUG:
            print(error_msg)
        return f'{{"response": "", "error": "{str(e)}"}}'