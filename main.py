# main.py (v1.2.3 - Fixed list() call and error handling)
import asyncio
import os
import uuid
import logging
import json
from typing import Dict, Any, AsyncGenerator, List, Optional

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
# --- Используем новый SDK ---
from google import genai
from google.genai import types as genai_types
# ИМПОРТИРУЕМ ошибки из genai
from google.genai import errors as genai_errors

from researcher import DeepResearcher, ResearchError

# --- Конфигурация ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gemini-1.5-pro-latest")

if not GEMINI_API_KEY:
    print("Ошибка: Переменная окружения GOOGLE_API_KEY не установлена.")
    print("Пожалуйста, создайте файл .env и добавьте GOOGLE_API_KEY='ВАШ_КЛЮЧ'")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Получение списка моделей Gemini ---
available_models: List[str] = []
try:
    client = genai.Client()
    # Используем config для page_size, как в документации
    models_list = client.models.list(config={'page_size': 100}) # Запросим побольше сразу

    available_models = sorted([
        m.name.replace("models/", "") for m in models_list
        if 'generateContent' in m.supported_actions
           and 'embedContent' not in m.supported_actions
           and 'tunedModels' not in m.name
    ])

    default_model_name_only = DEFAULT_MODEL_NAME.replace("models/", "")
    if default_model_name_only not in available_models:
        logger.warning(f"Модель по умолчанию '{default_model_name_only}' не найдена в списке доступных.")
        fallback_models = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-pro",
        ]
        found_fallback = False
        for model_name in fallback_models:
            model_name_only = model_name.replace("models/", "")
            if model_name_only in available_models:
                logger.warning(f"Используется fallback модель: '{model_name_only}'")
                DEFAULT_MODEL_NAME = model_name_only
                found_fallback = True
                break
        if not found_fallback:
            if available_models:
                 logger.warning(f"Fallback модели не найдены. Используется первая доступная: '{available_models[0]}'")
                 DEFAULT_MODEL_NAME = available_models[0]
            else:
                 logger.error("Не удалось получить список доступных моделей Gemini и найти fallback.")
                 available_models = [default_model_name_only]
                 DEFAULT_MODEL_NAME = default_model_name_only
    else:
         DEFAULT_MODEL_NAME = default_model_name_only

    logger.info(f"Доступные модели Gemini для генерации контента: {available_models}")
    logger.info(f"Модель по умолчанию: {DEFAULT_MODEL_NAME}")

# --- ИЗМЕНЕНИЯ В ОБРАБОТКЕ ОШИБОК ---
# Ловим только базовую ошибку API из genai.errors
except genai_errors.APIError as e:
     # Можно проверить e.code или e.message для специфики, если нужно
     if hasattr(e, 'code') and e.code == 401: # Пример проверки на PermissionDenied
          logger.exception(f"Ошибка прав доступа Google API (проверьте ключ): {e}")
     else:
          logger.exception(f"Ошибка Google API при получении списка моделей Gemini: {e}")
     exit(1)
except Exception as e:
    logger.exception(f"Критическая ошибка при получении списка моделей Gemini: {e}")
    exit(1)

# --- FastAPI Приложение ---
app = FastAPI(title="Deep Research Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Модели данных (без изменений) ---
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Основной запрос для исследования")
    depth: int = Field(default=2, ge=1, le=5, description="Глубина итераций исследования (1-5)")
    breadth: int = Field(default=3, ge=1, le=5, description="Ширина поиска на каждой итерации (1-5)")
    model_name: str = Field(default_factory=lambda: DEFAULT_MODEL_NAME, description="Имя модели Gemini для использования")

class ResearchTask(BaseModel):
    task_id: str
    status: str = "pending"
    progress_log: list[str] = []
    final_report: Optional[str] = None
    error: Optional[str] = None

# --- Хранилище задач (без изменений) ---
research_tasks: Dict[str, ResearchTask] = {}
task_events: Dict[str, asyncio.Queue] = {}

# --- Логика SSE (без изменений) ---
async def research_event_generator(task_id: str) -> AsyncGenerator[str, None]:
    queue = asyncio.Queue()
    task_events[task_id] = queue
    last_log_index = 0
    task_exists_checked = False

    try:
        while True:
            task = research_tasks.get(task_id)
            if not task and not task_exists_checked:
                await asyncio.sleep(1.0)
                task = research_tasks.get(task_id)
                if not task:
                    logger.warning(f"SSE: Задача {task_id} не найдена после ожидания.")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Задача не найдена или была удалена.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'status', 'content': 'failed'})}\n\n"
                    break
                task_exists_checked = True
            elif not task and task_exists_checked:
                logger.warning(f"SSE: Задача {task_id} была удалена во время стрима.")
                break

            if task:
                new_logs = task.progress_log[last_log_index:]
                for log_entry in new_logs:
                    try:
                        yield f"data: {json.dumps({'type': 'log', 'content': log_entry})}\n\n"
                    except TypeError as e:
                        logger.error(f"Ошибка сериализации лога в JSON для задачи {task_id}: {e} - Лог: {log_entry}")
                        yield f"data: {json.dumps({'type': 'log', 'content': f'[Ошибка сериализации лога: {e}]'})}\n\n"
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
            except Exception as e:
                 logger.error(f"Неожиданная ошибка при ожидании очереди SSE для задачи {task_id}: {e}")
                 break

    except asyncio.CancelledError:
        logger.info(f"SSE соединение закрыто клиентом для задачи {task_id}")
    except Exception as e:
        logger.exception(f"Критическая ошибка в генераторе SSE для задачи {task_id}: {e}")
        try:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Внутренняя ошибка сервера SSE.'})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'content': 'failed'})}\n\n"
        except:
            pass
    finally:
        if task_id in task_events:
            del task_events[task_id]
            logger.debug(f"Очередь событий удалена для задачи {task_id}")
        # asyncio.create_task(cleanup_task(task_id, delay=600))

async def cleanup_task(task_id: str, delay: int):
    """Удаляет задачу из памяти после задержки."""
    await asyncio.sleep(delay)
    if task_id in research_tasks:
        del research_tasks[task_id]
        logger.info(f"Задача {task_id} удалена из памяти по таймеру.")
    if task_id in task_events:
        del task_events[task_id]

# --- Фоновая задача исследования ---
async def run_research_task(task_id: str, query: str, depth: int, breadth: int, model_name: str):
    """Выполняет исследование в фоновом режиме и обновляет статус задачи."""
    task = research_tasks.get(task_id)
    if not task:
        logger.error(f"Не удалось найти задачу {task_id} для запуска исследования.")
        return

    task.status = "running"
    task.progress_log.append(f"Исследование запущено (Модель: {model_name})...")
    if task_id in task_events:
        task_events[task_id].put_nowait("update")

    try:
        researcher = DeepResearcher(
            model_name=model_name,
            depth=depth,
            breadth=breadth,
            max_completion_tokens=8000,
            temperature=0.3,
            request_timeout=300
        )

        async def log_callback_async(log_message: str):
             task.progress_log.append(log_message)
             if task_id in task_events:
                 try:
                     task_events[task_id].put_nowait("update")
                 except asyncio.QueueFull:
                      logger.warning(f"Очередь SSE для задачи {task_id} переполнена.")
                 except Exception as e:
                      logger.error(f"Ошибка уведомления SSE для задачи {task_id}: {e}")

        final_report = await researcher.research(query, log_callback=log_callback_async)

        task.final_report = final_report
        task.status = "completed"
        task.progress_log.append("Исследование успешно завершено.")

    except ResearchError as e:
        logger.error(f"Ошибка исследования (ResearchError) для задачи {task_id}: {e}")
        task.status = "failed"
        task.error = str(e)
        task.progress_log.append(f"Ошибка исследования: {e}")
    # --- ИЗМЕНЕНИЯ В ОБРАБОТКЕ ОШИБОК ---
    except genai_errors.APIError as e: # Ловим базовую ошибку genai
        logger.error(f"Ошибка Google API в задаче {task_id}: {e}")
        task.status = "failed"
        error_message = str(e)
        if hasattr(e, 'message'):
             error_message = e.message
        task.error = f"Ошибка взаимодействия с Google API: {error_message}"
        task.progress_log.append(f"Ошибка Google API: {error_message}")
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка в задаче {task_id}: {e}")
        task.status = "failed"
        task.error = "Внутренняя ошибка сервера при выполнении исследования."
        task.progress_log.append(f"Критическая непредвиденная ошибка: {e}")
    finally:
        if task_id in task_events:
             try:
                task_events[task_id].put_nowait("update")
             except asyncio.QueueFull:
                 logger.warning(f"Очередь SSE для задачи {task_id} переполнена при финальном уведомлении.")
             except Exception as e:
                 logger.error(f"Ошибка финального уведомления SSE для задачи {task_id}: {e}")

# --- Эндпоинты FastAPI (без изменений) ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "available_models": available_models,
        "default_model": DEFAULT_MODEL_NAME
    })

@app.post("/research", status_code=202)
async def start_research(
    request: ResearchRequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    model_name_only = request.model_name.replace("models/", "")
    if model_name_only not in available_models:
         logger.warning(f"Запрошена недоступная модель: {model_name_only}. Доступные: {available_models}")
         raise HTTPException(status_code=400, detail=f"Выбранная модель '{model_name_only}' недоступна или не поддерживает генерацию контента.")

    task_id = str(uuid.uuid4())
    research_tasks[task_id] = ResearchTask(task_id=task_id, status="pending")
    logger.info(f"Создана новая задача исследования: {task_id} для запроса: '{request.query}' с моделью '{model_name_only}'")

    background_tasks.add_task(
        run_research_task,
        task_id,
        request.query,
        request.depth,
        request.breadth,
        model_name_only
    )

    return {"task_id": task_id, "message": "Исследование запущено"}

@app.get("/research/stream/{task_id}")
async def stream_research_progress(task_id: str):
    if task_id not in research_tasks:
        await asyncio.sleep(0.5)
    if task_id not in research_tasks:
         logger.error(f"SSE запрос для несуществующей задачи: {task_id}")
         raise HTTPException(status_code=404, detail="Задача не найдена")

    logger.info(f"Открыто SSE соединение для задачи {task_id}")
    return EventSourceResponse(research_event_generator(task_id))

@app.get("/research/status/{task_id}")
async def get_research_status(task_id: str) -> ResearchTask:
    task = research_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task
