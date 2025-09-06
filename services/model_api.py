import asyncio
import aiohttp
import tenacity
import json

# ==== API вызовы ====
class ModelAPIError(Exception):
    pass

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    retry=tenacity.retry_if_exception_type((aiohttp.ClientError, ModelAPIError))
)
async def get_completion(url, token, messages, model, temperature=0.7):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(
            url=url,
            json={"model": model, "messages": messages, "temperature": temperature}
        ) as resp:
            if resp.status != 200:
                raise ModelAPIError(f"Ошибка API {resp.status}: {await resp.text()}")
            response_json = await resp.json()
            return response_json["choices"][0]["message"]["content"]

async def wrapped_get_completion(*args, **kwargs):
    try:
        return await get_completion(*args, **kwargs)
    except Exception as e:
        return f"[Ошибка LLM]: {str(e)}"