# main.py (v1.1.1)
import asyncio
import os
import uuid
import logging
import json # <-- ИСПРАВЛЕНИЕ: Добавлен импорт json
from typing import Dict, Any, AsyncGenerator, List

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import google.generativeai as genai

from researcher import DeepResearcher, ResearchError

# --- Конфигурация ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gemini-2.5-pro-preview-03-25")

if not GEMINI_API_KEY:
    print("Ошибка: Переменная окружения GEMINI_API_KEY не установлена.")
    print("Пожалуйста, создайте файл .env и добавьте GEMINI_API_KEY='ВАШ_КЛЮЧ'")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Получение списка моделей Gemini ---
available_models: List[str] = []
try:
    genai.configure(api_key=GEMINI_API_KEY)
    models_list = genai.list_models()
    available_models = sorted([
        m.name.replace("models/", "") for m in models_list
        if 'generateContent' in m.supported_generation_methods and 'embedding' not in m.name
    ])
    if DEFAULT_MODEL_NAME not in available_models and available_models:
        logger.warning(f"Модель по умолчанию '{DEFAULT_MODEL_NAME}' не найдена в доступных. Используется первая доступная: '{available_models[0]}'")
        DEFAULT_MODEL_NAME = available_models[0]
    elif not available_models:
         logger.error("Не удалось получить список доступных моделей Gemini.")
         available_models = [DEFAULT_MODEL_NAME]
    logger.info(f"Доступные модели Gemini: {available_models}")
    logger.info(f"Модель по умолчанию: {DEFAULT_MODEL_NAME}")

except Exception as e:
    logger.exception(f"Критическая ошибка при получении списка моделей Gemini: {e}")
    exit(1)

# --- FastAPI Приложение ---
app = FastAPI(title="Deep Research Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Модели данных ---
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Основной запрос для исследования")
    depth: int = Field(default=2, ge=1, le=5, description="Глубина итераций исследования (1-5)")
    breadth: int = Field(default=3, ge=1, le=5, description="Ширина поиска на каждой итерации (1-5)")
    model_name: str = Field(default=DEFAULT_MODEL_NAME, description="Имя модели Gemini для использования")

class ResearchTask(BaseModel):
    task_id: str
    status: str = "pending"
    progress_log: list[str] = []
    final_report: str | None = None
    error: str | None = None

# --- Хранилище задач (в памяти) ---
research_tasks: Dict[str, ResearchTask] = {}
task_events: Dict[str, asyncio.Queue] = {}

# --- Логика SSE ---
async def research_event_generator(task_id: str) -> AsyncGenerator[str, None]:
    """Генератор Server-Sent Events для стриминга прогресса исследования."""
    queue = asyncio.Queue()
    task_events[task_id] = queue
    last_log_index = 0

    try:
        while True:
            task = research_tasks.get(task_id)
            if not task:
                await asyncio.sleep(0.5)
                continue

            new_logs = task.progress_log[last_log_index:]
            for log_entry in new_logs:
                # ИСПРАВЛЕНИЕ: Используем импортированный json
                yield f"data: {json.dumps({'type': 'log', 'content': log_entry})}\n\n"
            last_log_index = len(task.progress_log)

            if task.status == "completed":
                yield f"data: {json.dumps({'type': 'report', 'content': task.final_report})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'content': 'completed'})}\n\n"
                break
            elif task.status == "failed":
                yield f"data: {json.dumps({'type': 'error', 'content': task.error})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'content': 'failed'})}\n\n"
                break

            try:
                await asyncio.wait_for(queue.get(), timeout=30)
                queue.task_done()
            except asyncio.TimeoutError:
                pass

    except asyncio.CancelledError:
        logger.info(f"SSE соединение закрыто для задачи {task_id}")
    finally:
        if task_id in task_events:
            del task_events[task_id]
        # asyncio.create_task(cleanup_task(task_id, delay=600))

async def cleanup_task(task_id: str, delay: int):
    """Удаляет задачу из памяти после задержки."""
    await asyncio.sleep(delay)
    if task_id in research_tasks:
        del research_tasks[task_id]
        logger.info(f"Задача {task_id} удалена из памяти.")

# --- Фоновая задача исследования ---
async def run_research_task(task_id: str, query: str, depth: int, breadth: int, model_name: str):
    """Выполняет исследование в фоновом режиме и обновляет статус задачи."""
    task = research_tasks[task_id]
    task.status = "running"
    task.progress_log.append(f"Исследование запущено (Модель: {model_name})...")
    if task_id in task_events:
        task_events[task_id].put_nowait("update")

    try:
        researcher = DeepResearcher(
            api_key=GEMINI_API_KEY,
            depth=depth,
            breadth=breadth,
            model_name=model_name,
            max_completion_tokens=8000
        )

        # Определяем колбэк внутри этой асинхронной функции
        async def log_callback_async(log_message: str):
             task.progress_log.append(log_message)
             if task_id in task_events:
                 try:
                     task_events[task_id].put_nowait("update")
                 except Exception as e:
                      logger.error(f"Ошибка уведомления SSE для задачи {task_id}: {e}")

        final_report = await researcher.research(query, log_callback=log_callback_async) # Передаем асинхронный колбэк

        task.final_report = final_report
        task.status = "completed"
        task.progress_log.append("Исследование завершено.")

    except ResearchError as e:
        logger.error(f"Ошибка исследования для задачи {task_id}: {e}")
        task.status = "failed"
        task.error = str(e)
        task.progress_log.append(f"Ошибка: {e}")
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка в задаче {task_id}: {e}")
        task.status = "failed"
        task.error = "Внутренняя ошибка сервера."
        task.progress_log.append(f"Критическая ошибка: {e}")
    finally:
        if task_id in task_events:
             try:
                task_events[task_id].put_nowait("update")
             except Exception as e:
                 logger.error(f"Ошибка финального уведомления SSE для задачи {task_id}: {e}")


# --- Эндпоинты FastAPI ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Отдает главную HTML страницу со списком моделей."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "available_models": available_models,
        "default_model": DEFAULT_MODEL_NAME
    })

@app.post("/research", status_code=202)
async def start_research(
    request: ResearchRequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Запускает новую задачу исследования в фоновом режиме."""
    if request.model_name not in available_models:
         raise HTTPException(status_code=400, detail=f"Выбранная модель '{request.model_name}' недоступна.")

    task_id = str(uuid.uuid4())
    research_tasks[task_id] = ResearchTask(task_id=task_id, status="pending")
    logger.info(f"Создана новая задача исследования: {task_id} для запроса: '{request.query}' с моделью '{request.model_name}'")

    background_tasks.add_task(
        run_research_task,
        task_id,
        request.query,
        request.depth,
        request.breadth,
        request.model_name
    )

    return {"task_id": task_id, "message": "Исследование запущено"}

@app.get("/research/stream/{task_id}")
async def stream_research_progress(task_id: str):
    """Открывает SSE соединение для стриминга прогресса задачи."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    logger.info(f"Открыто SSE соединение для задачи {task_id}")
    return EventSourceResponse(research_event_generator(task_id))

@app.get("/research/status/{task_id}")
async def get_research_status(task_id: str) -> ResearchTask:
    """Возвращает текущий статус и результат задачи (альтернатива SSE)."""
    task = research_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task

# --- Запуск Uvicorn (для локальной разработки) ---
# Блок if __name__ == "__main__": убран,
# так как Gunicorn сам импортирует и запускает 'main:app'
