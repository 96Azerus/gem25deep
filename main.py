# main.py (v1.2.0 - Migrated to google-genai)
import asyncio
import os
import uuid
import logging
import json
from typing import Dict, Any, AsyncGenerator, List

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
# --- ИЗМЕНЕНИЯ ИМПОРТОВ ---
from google import genai # Используем новый SDK
from google.genai import types as genai_types # Для типов ошибок и т.д., если нужно
from google.api_core import exceptions as google_api_exceptions # Для обработки ошибок API

from researcher import DeepResearcher, ResearchError

# --- Конфигурация ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # <-- ИЗМЕНЕНО: Новый SDK ищет GOOGLE_API_KEY
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gemini-1.5-pro-latest") # Обновил дефолтную модель

if not GEMINI_API_KEY:
    print("Ошибка: Переменная окружения GOOGLE_API_KEY не установлена.")
    print("Пожалуйста, создайте файл .env и добавьте GOOGLE_API_KEY='ВАШ_КЛЮЧ'")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Получение списка моделей Gemini (с использованием нового SDK) ---
available_models: List[str] = []
try:
    # Настройка клиента происходит при его создании или через env var
    # genai.configure(api_key=GEMINI_API_KEY) # <-- УБРАНО
    client = genai.Client(api_key=GEMINI_API_KEY) # Создаем клиент для проверки и получения списка
    models_list = client.models.list() # Используем метод клиента

    available_models = sorted([
        m.name.replace("models/", "") for m in models_list
        # Проверяем поддержку 'generateContent' - может называться иначе или быть неявной
        # В новом SDK лучше фильтровать по имени или явно известным моделям
        # if 'generateContent' in m.supported_generation_methods # <-- УБРАНО/ИЗМЕНЕНО
        if 'generateContent' in m.supported_actions # Проверяем по supported_actions
           and 'embedContent' not in m.supported_actions # Исключаем модели только для эмбеддингов
           and 'tunedModels' not in m.name # Исключаем явно тюнингованные модели из общего списка
    ])

    # Проверяем, существует ли модель по умолчанию в списке доступных
    if DEFAULT_MODEL_NAME not in available_models:
        # Логика Fallback остается прежней, но можно обновить список моделей
        fallback_models = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-pro", # Добавим базовую gemini-pro
            # Старые preview модели могут быть уже недоступны
        ]
        found_fallback = False
        for model in fallback_models:
            # Имена моделей в новом SDK могут быть без префикса 'models/' при листинге,
            # но требовать его при вызове. Приводим к единому виду для сравнения.
            model_name_only = model.replace("models/", "")
            if model_name_only in available_models:
                logger.warning(f"Модель по умолчанию '{DEFAULT_MODEL_NAME}' не найдена. Используется fallback: '{model_name_only}'")
                DEFAULT_MODEL_NAME = model_name_only
                found_fallback = True
                break
        # Если даже fallback не найден, берем первую из списка
        if not found_fallback and available_models:
             logger.warning(f"Модель по умолчанию '{DEFAULT_MODEL_NAME}' и fallback модели не найдены. Используется первая доступная: '{available_models[0]}'")
             DEFAULT_MODEL_NAME = available_models[0]
        elif not available_models:
             logger.error("Не удалось получить список доступных моделей Gemini. Используется жестко заданная модель по умолчанию.")
             # Добавляем модель по умолчанию в список, чтобы она была доступна для выбора
             available_models = [DEFAULT_MODEL_NAME]

    logger.info(f"Доступные модели Gemini для генерации контента: {available_models}")
    logger.info(f"Модель по умолчанию: {DEFAULT_MODEL_NAME}")

# Обрабатываем специфичные ошибки API Google и общие ошибки
except google_api_exceptions.GoogleAPIError as e:
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
    model_name: str = Field(default=DEFAULT_MODEL_NAME, description="Имя модели Gemini для использования")

class ResearchTask(BaseModel):
    task_id: str
    status: str = "pending"
    progress_log: list[str] = []
    final_report: str | None = None
    error: str | None = None

# --- Хранилище задач (в памяти) - ОСТАЕТСЯ ПРОБЛЕМОЙ ДЛЯ ПРОДА ---
research_tasks: Dict[str, ResearchTask] = {}
task_events: Dict[str, asyncio.Queue] = {}

# --- Логика SSE (без изменений в логике отправки, только обработка ошибок) ---
async def research_event_generator(task_id: str) -> AsyncGenerator[str, None]:
    """Генератор Server-Sent Events для стриминга прогресса исследования."""
    queue = asyncio.Queue()
    task_events[task_id] = queue
    last_log_index = 0

    try:
        while True:
            task = research_tasks.get(task_id)
            if not task:
                # Добавим небольшую паузу, чтобы не грузить CPU в ожидании задачи
                await asyncio.sleep(0.5)
                continue # Задача еще не создана или уже удалена

            # Отправляем новые логи
            new_logs = task.progress_log[last_log_index:]
            for log_entry in new_logs:
                try:
                    yield f"data: {json.dumps({'type': 'log', 'content': log_entry})}\n\n"
                except TypeError as e:
                     logger.error(f"Ошибка сериализации лога в JSON для задачи {task_id}: {e} - Лог: {log_entry}")
                     # Можно отправить сообщение об ошибке или пропустить лог
                     yield f"data: {json.dumps({'type': 'log', 'content': f'[Ошибка сериализации лога: {e}]'})}\n\n"

            last_log_index = len(task.progress_log)

            # Проверяем статус завершения
            if task.status == "completed":
                yield f"data: {json.dumps({'type': 'report', 'content': task.final_report})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'content': 'completed'})}\n\n"
                break
            elif task.status == "failed":
                yield f"data: {json.dumps({'type': 'error', 'content': task.error})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'content': 'failed'})}\n\n"
                break

            # Ожидаем обновлений от фоновой задачи
            try:
                # Ждем сигнала из очереди или таймаута
                await asyncio.wait_for(queue.get(), timeout=30)
                queue.task_done()
            except asyncio.TimeoutError:
                # Таймаут - просто продолжаем цикл, чтобы проверить статус задачи снова
                pass
            except Exception as e:
                 logger.error(f"Неожиданная ошибка при ожидании очереди SSE для задачи {task_id}: {e}")
                 # Решаем, прерывать ли цикл или продолжать
                 break # Прерываем в случае неизвестной ошибки

    except asyncio.CancelledError:
        logger.info(f"SSE соединение закрыто клиентом для задачи {task_id}")
    except Exception as e:
        logger.exception(f"Критическая ошибка в генераторе SSE для задачи {task_id}: {e}")
        # Попытка отправить сообщение об ошибке, если возможно
        try:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Внутренняя ошибка сервера SSE.'})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'content': 'failed'})}\n\n"
        except:
            pass # Если отправить не удалось, просто выходим
    finally:
        if task_id in task_events:
            del task_events[task_id]
            logger.debug(f"Очередь событий удалена для задачи {task_id}")
        # Раскомментируем и используем, если нужна очистка задач
        # asyncio.create_task(cleanup_task(task_id, delay=600)) # 10 минут

async def cleanup_task(task_id: str, delay: int):
    """Удаляет задачу из памяти после задержки."""
    await asyncio.sleep(delay)
    if task_id in research_tasks:
        del research_tasks[task_id]
        logger.info(f"Задача {task_id} удалена из памяти по таймеру.")
    if task_id in task_events: # На всякий случай, если генератор завершился некорректно
        del task_events[task_id]

# --- Фоновая задача исследования (обновлена обработка ошибок) ---
async def run_research_task(task_id: str, query: str, depth: int, breadth: int, model_name: str):
    """Выполняет исследование в фоновом режиме и обновляет статус задачи."""
    task = research_tasks[task_id]
    task.status = "running"
    # Используем полное имя модели, как оно в списке available_models
    task.progress_log.append(f"Исследование запущено (Модель: {model_name})...")
    if task_id in task_events:
        task_events[task_id].put_nowait("update") # Уведомляем SSE о начале

    try:
        # Инициализация DeepResearcher теперь не требует API ключа здесь,
        # он должен быть доступен как переменная окружения GOOGLE_API_KEY
        researcher = DeepResearcher(
            # api_key=GEMINI_API_KEY, # <-- УБРАНО: Клиент создается внутри
            depth=depth,
            breadth=breadth,
            model_name=model_name, # Передаем имя модели для использования в вызовах
            max_completion_tokens=8000 # Можно сделать настраиваемым
        )

        async def log_callback_async(log_message: str):
             task.progress_log.append(log_message)
             if task_id in task_events:
                 try:
                     # Используем put_nowait, так как блокировка здесь нежелательна
                     task_events[task_id].put_nowait("update")
                 except asyncio.QueueFull:
                      logger.warning(f"Очередь SSE для задачи {task_id} переполнена. Лог может быть пропущен.")
                 except Exception as e:
                      logger.error(f"Ошибка уведомления SSE для задачи {task_id}: {e}")

        # Запускаем исследование
        final_report = await researcher.research(query, log_callback=log_callback_async)

        # Обработка результата
        task.final_report = final_report
        task.status = "completed"
        task.progress_log.append("Исследование успешно завершено.")

    except ResearchError as e:
        logger.error(f"Ошибка исследования (ResearchError) для задачи {task_id}: {e}")
        task.status = "failed"
        task.error = str(e) # Используем сообщение из ResearchError
        task.progress_log.append(f"Ошибка исследования: {e}")
    except google_api_exceptions.GoogleAPIError as e:
        logger.error(f"Ошибка Google API в задаче {task_id}: {e}")
        task.status = "failed"
        # Предоставляем более конкретную ошибку API, если возможно
        task.error = f"Ошибка взаимодействия с Google API: {e.message}"
        task.progress_log.append(f"Ошибка Google API: {e.message}")
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка в задаче {task_id}: {e}")
        task.status = "failed"
        task.error = "Внутренняя ошибка сервера при выполнении исследования."
        task.progress_log.append(f"Критическая непредвиденная ошибка: {e}")
    finally:
        # Финальное уведомление SSE для обновления статуса и отчета/ошибки
        if task_id in task_events:
             try:
                task_events[task_id].put_nowait("update")
             except asyncio.QueueFull:
                 logger.warning(f"Очередь SSE для задачи {task_id} переполнена при финальном уведомлении.")
             except Exception as e:
                 logger.error(f"Ошибка финального уведомления SSE для задачи {task_id}: {e}")

# --- Эндпоинты FastAPI (без существенных изменений) ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Отдает главную HTML страницу со списком моделей."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "available_models": available_models, # Передаем отфильтрованный список
        "default_model": DEFAULT_MODEL_NAME
    })

@app.post("/research", status_code=202)
async def start_research(
    request: ResearchRequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Запускает новую задачу исследования в фоновом режиме."""
    # Проверяем доступность модели по списку, полученному при старте
    if request.model_name not in available_models:
         raise HTTPException(status_code=400, detail=f"Выбранная модель '{request.model_name}' недоступна или не поддерживает генерацию контента.")

    task_id = str(uuid.uuid4())
    research_tasks[task_id] = ResearchTask(task_id=task_id, status="pending")
    logger.info(f"Создана новая задача исследования: {task_id} для запроса: '{request.query}' с моделью '{request.model_name}'")

    background_tasks.add_task(
        run_research_task,
        task_id,
        request.query,
        request.depth,
        request.breadth,
        request.model_name # Передаем имя модели в фоновую задачу
    )

    return {"task_id": task_id, "message": "Исследование запущено"}

@app.get("/research/stream/{task_id}")
async def stream_research_progress(task_id: str):
    """Открывает SSE соединение для стриминга прогресса задачи."""
    if task_id not in research_tasks:
        # Можно добавить небольшую задержку и повторную проверку,
        # если клиент подключается очень быстро после POST /research
        await asyncio.sleep(0.5)
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
# Блок if __name__ == "__main__": убран для совместимости с Gunicorn
