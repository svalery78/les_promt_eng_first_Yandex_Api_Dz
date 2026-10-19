import openai
import os
import sys
from pathlib import Path

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
- Ты — организатор досуга.
- Предложи ровно 3 разных варианта, как провести вечер, учитывая интересы,
  место и ограничения из сообщения пользователя. Отвечай на русском языке.
- Для каждого варианта укажи название и короткий план из 2–3 действий.
- Не придумывай сведения о пользователе, цены и доступность мероприятий.
- Не добавляй длинное вступление. Ориентир — до 250 слов.
    """

TEMPERATURE = 0.8

DEFAULT_QUERY = "Хочу отдохнуть дома, люблю кино и вкусную еду"

def extract_text(response) -> str:
    """Извлекает текст и из стандартного ответа SDK, и из content-блоков."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
    return "\n".join(parts).strip()

def generate_evening(client, model: str, query: str, temperature: float):
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=query,
        temperature=temperature,
        max_output_tokens=1800,
    )
    status = getattr(response, "status", None)
    if status and status != "completed":
        raise RuntimeError(
            f"Ответ не завершён: status={status}. "
            "Проверьте статус запроса и лимит max_output_tokens."
        )
    text = extract_text(response)
    if not text:
        raise RuntimeError("API не вернул текст ответа.")
    return text, response

def main() -> int:
    directory = Path(__file__).resolve().parent
    load_dotenv(directory / ".env")
    api_key = os.getenv("YANDEX_API_KEY", "").strip()
    folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
    if not api_key or not folder_id:
        print("Укажите YANDEX_API_KEY и YANDEX_FOLDER_ID в .env рядом со скриптом.", file=sys.stderr)
        return 1
    model_name = os.getenv("YANDEX_MODEL", "yandexgpt-5-lite").strip()
    if not model_name:
        print("YANDEX_MODEL не должен быть пустым.", file=sys.stderr)
        return 1
    model = f"gpt://{folder_id}/{model_name}"
    try:
        query = input(f"Запрос (Enter — {DEFAULT_QUERY}): ").strip() or DEFAULT_QUERY
    except (EOFError, KeyboardInterrupt):
        print("\nВвод отменён.", file=sys.stderr)
        return 1

    with openai.OpenAI(
        api_key=api_key,
        base_url="https://ai.api.cloud.yandex.net/v1",
        project=folder_id,
        timeout=60.0,
        max_retries=0,
    ) as client:
        print(f"\nЗапрос к модели: temperature={TEMPERATURE}...")
        text, _ = generate_evening(client, model, query, TEMPERATURE)
        print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except openai.APIStatusError as exc:
        print(f"Ошибка API: HTTP {exc.status_code}. Проверьте ключ, права, модель и баланс.", file=sys.stderr)
        sys.exit(1)
    except openai.APIConnectionError:
        print("Не удалось получить ответ API: проверьте соединение и доступность сервиса.", file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)