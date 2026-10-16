import openai
import os

from dotenv import load_dotenv

load_dotenv(override=True)

YANDEX_MODEL = "yandexgpt-5-lite"
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

if not YANDEX_API_KEY:
        raise ValueError("Не найден YANDEX_API_KEY в .env")

if not YANDEX_FOLDER_ID:
        raise ValueError("Не найден YANDEX_FOLDER_ID в .env")

SYSTEM_PROMPT = """
    _Ты — ассистент-ресерчер, который ищет в интернете актуальные материалы и полезные ресурсы._
    
    _Ты специализируешься на поиске свежих статей, новостей, исследований и инструментов._
    
    _**Правила:**_
    
    1. _Всегда сначала используй интернет-поиск._
    
    2. _Отдавай приоритет свежим материалам._
    
    3. _Отдавай приоритет авторитетным источникам: официальные сайты, научные публикации, GitHub, документация._
    
    4. _Не придумывай ссылки — используй только найденные материалы._
    
    5. _В ответе всегда давай ровно 5 ссылок._
    
    6. _Для каждой ссылки укажи:_
        
        - _название материала;_
        
        - _URL;_
        
        - _1–2 предложения, почему источник полезен._

    7. _Важно: всегда отвечай только на русском языке._
    
        - _Даже если источники на английском — переводи заголовки и описания на русский;_
    
        - _Не используй английский язык в тексте ответа, кроме оригинальных названий продуктов, компаний и ссылок;_
    
        - _Не переключайся на английский ни при каких условиях_    
    
    8. _Если прямых материалов мало, расширяй поиск на англоязычные источники, но всё равно дай 5 ссылок._
    
    9. _Сначала дай короткий вывод (1–2 предложения), затем список ссылок._
    """

client = openai.OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1",
    project= YANDEX_FOLDER_ID
)

def research(query: str) -> str:    
    response = client.responses.create(
        model=f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}",
        instructions = SYSTEM_PROMPT,
        input=query,
        tools=[
            {
                "type": "web_search"
            }
        ],
        temperature=0.2,
        max_output_tokens=1800
    )

    text_parts = []

    if hasattr(response, "output") and response.output:
        for item in response.output:
            if hasattr(item, "content") and item.content:
                for content_item in item.content:
                    text = getattr(content_item, "text", None)
                    if text:
                        text_parts.append(text)

    return "\n".join(text_parts).strip()

if __name__ == "__main__":
    user_query = "Найди свежие новости и полезные материалы про AI agents и web search в LLM"
    result = research(user_query)
    print(result)